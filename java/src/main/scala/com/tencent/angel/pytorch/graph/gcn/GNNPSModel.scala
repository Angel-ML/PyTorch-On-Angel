/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.exception.InvalidParameterException
import com.tencent.angel.graph.client.psf.get.getnodes.{GetNodes, GetNodesParam}
import com.tencent.angel.graph.client.psf.get.getlabels.GetLabels
import com.tencent.angel.graph.client.psf.get.getnodefeats.{GetNodeFeats, GetNodeFeatsResult}
import com.tencent.angel.graph.client.psf.get.utils.{GetFloatArrayAttrsResult, GetNodeAttrsParam}
import com.tencent.angel.graph.client.psf.init.GeneralInitParam
import com.tencent.angel.graph.client.psf.init.initedgefeats.InitEdgeFeats
import com.tencent.angel.graph.client.psf.init.initlabels.InitLabels
import com.tencent.angel.graph.client.psf.init.initneighbors.InitNeighbor
import com.tencent.angel.graph.client.psf.init.initnodefeats.InitNodeFeats
import com.tencent.angel.graph.client.psf.init.initnodetypes.InitNodeTypes
import com.tencent.angel.graph.client.psf.sample.samplenodefeats._
import com.tencent.angel.graph.client.psf.sample.sampleedgefeats._
import com.tencent.angel.graph.client.psf.sample.sampleneighbor._
import com.tencent.angel.graph.client.psf.summary.{NnzEdge, NnzFeature, NnzNeighbor, NnzNode}
import com.tencent.angel.graph.common.param.ModelContext
import com.tencent.angel.graph.data._
import com.tencent.angel.graph.utils.ModelContextUtils
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.vector.{IntFloatVector, IntLongVector, LongFloatVector}
import com.tencent.angel.ml.matrix.psf.aggr.Size
import com.tencent.angel.ml.matrix.psf.aggr.enhance.ScalarAggrResult
import com.tencent.angel.ml.matrix.psf.get.getrow.GetRowResult
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.ps.storage.vector.element.IElement
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.pytorch.utils.CheckpointUtils
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import it.unimi.dsi.fastutil.ints.{IntArrayList, IntArrays}
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

private[gcn]
class GNNPSModel(graph: PSMatrix,
                 weights: PSVector,
                 labels: PSVector,
                 testLabels: PSVector) extends Serializable {

  val dim: Long = labels.dimension

  // the default pull method will return keys even those not exists on servers
  def readLabels(keys: Array[Long]): LongFloatVector =
    labels.pull(keys.clone()).asInstanceOf[LongFloatVector]

  // this method will not return keys that do not exist on servers
  def readLabels2(keys: Array[Long]): LongFloatVector = {
    import com.tencent.angel.graph.psf.gcn.{GetLabels, GetLabelsResult}
    val func = new GetLabels(labels.poolId, keys.clone())
    labels.psfGet(func).asInstanceOf[GetLabelsResult].getVector
  }

  def readTestLabels(keys: Array[Long]): LongFloatVector = {
    import com.tencent.angel.graph.psf.gcn.{GetLabels, GetLabelsResult}
    val func = new GetLabels(testLabels.poolId, keys.clone())
    testLabels.psfGet(func).asInstanceOf[GetLabelsResult].getVector
  }

  def setLabels(value: LongFloatVector): Unit =
    labels.update(value)

  def setTestLabels(value: LongFloatVector): Unit =
    testLabels.update(value)

  def nnzLabels(): Long =
    labels.psfGet(new Size(labels.poolId, labels.id)).asInstanceOf[ScalarAggrResult].getResult.toLong

  def nnzTestLabels(): Long =
    testLabels.psfGet(new Size(testLabels.poolId, testLabels.id)).asInstanceOf[ScalarAggrResult].getResult.toLong

  def readWeights(): Array[Float] =
    weights.pull().asInstanceOf[IntFloatVector].getStorage.getValues

  def setWeights(values: Array[Float]): Unit = {
    val update = VFactory.denseFloatVector(values)
    weights.update(update)
  }

  def step(grads: Array[Float], optim: AsyncOptim): Unit = {
    val update = VFactory.denseFloatVector(grads)
    optim.asyncUpdate(weights, 1, update).get()
  }

  /* summary functions */
  def nnzNodes(): Long = {
    val func = new NnzNode(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def nnzNeighbors(): Long = {
    val func = new NnzNeighbor(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def nnzFeatures(): Long = {
    val func = new NnzFeature(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def nnzEdge(): Long = {
    val func = new NnzEdge(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  // if use multi-label classification, init label (Array[Float]) to Node labels,
  // in which the first value indicates training or testing node

  def initMultiLabelsByBatch(pairs: Seq[(Long, Array[Float])]): Int = {
    val nodeIds = new Array[Long](pairs.size)
    val labels = new Array[IElement](pairs.size)

    pairs.zipWithIndex.foreach(elem => {
      nodeIds(elem._2) = elem._1._1
      labels(elem._2) = new Labels(elem._1._2)
    })

    val func = new InitLabels(new GeneralInitParam(graph.id, nodeIds, labels))
    graph.asyncPsfUpdate(func).get()
    println(s"init ${pairs.length} labels")
    pairs.length
  }

  def readMultiLabels(nodeIds: Array[Long]): Long2ObjectOpenHashMap[Array[Float]] = {
    graph.psfGet(new GetLabels(new GetNodeAttrsParam(graph.id, nodeIds)))
      .asInstanceOf[GetFloatArrayAttrsResult].getNodeIdToContents
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    numBatch: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    var splitStart = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      val indptrBatch = indptr.slice(start + 1, end + 1)
      val neighborsBatch = new Array[Array[Long]](indptrBatch.length)
      for (i <- indptrBatch.indices) {
        neighborsBatch(i) = neighbors.slice(splitStart, indptrBatch(i))
        splitStart = indptrBatch(i)
      }
      initNeighborsByBatch(keys.slice(start, end), neighborsBatch)
      start += step
    }
  }

  def initNeighborsByBatch(batchKeys: Array[Long], batchNeighbors: Array[Array[Long]]): Unit = {
    val neighbors = new Array[IElement](batchKeys.length)
    for (i <- batchNeighbors.indices) {
      neighbors(i) = new LongNeighbor(batchNeighbors(i))
    }
    val func = new InitNeighbor(new GeneralInitParam(graph.id, batchKeys, neighbors))
    graph.asyncPsfUpdate(func).get()
    println(s"init ${batchKeys.length} neighbors")
  }

  def initNeighborsByBatch(batchKeys: Array[Long], batchNeighbors: Array[Array[Long]],
                           batchTypes: Array[Array[Int]]): Unit = {
    val neighbors = new Array[IElement](batchKeys.length)
    for (i <- batchNeighbors.indices) {
      neighbors(i) = new LongNeighbor(batchNeighbors(i))
    }
    val func = new InitNeighbor(new GeneralInitParam(graph.id, batchKeys.clone(), neighbors))
    graph.asyncPsfUpdate(func).get()
    println(s"init ${batchKeys.length} neighbors")

    val types = new Array[IElement](batchKeys.length)
    for (i <- batchTypes.indices) {
      types(i) = new NodeType(batchTypes(i))
    }
    val nodeFunc = new InitNodeTypes(new GeneralInitParam(graph.id, batchKeys.clone(), types))
    graph.asyncPsfUpdate(nodeFunc).get()
    println(s"init ${batchKeys.length} node types")
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    numBatch: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    var splitStart = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      val indptrBatch = indptr.slice(start + 1, end + 1)
      val neighborsBatch = new Array[Array[Long]](indptrBatch.length)
      val typesBatch = new Array[Array[Int]](indptrBatch.length)
      for (i <- indptrBatch.indices) {
        neighborsBatch(i) = neighbors.slice(splitStart, indptrBatch(i))
        typesBatch(i) = types.slice(splitStart, indptrBatch(i))
        splitStart = indptrBatch(i)
      }
      initNeighborsByBatch(keys.slice(start, end), neighborsBatch, typesBatch)
      start += step
    }
  }

  def initEdgeFeatures(keys: Array[Long],
                       indptr: Array[Int],
                       neighbors: Array[Long],
                       features: Array[IntFloatVector],
                       numBatch: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    var splitStart = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      val indptrBatch = indptr.slice(start + 1, end + 1)
      val neighborsBatch = new Array[Array[Long]](indptrBatch.length)
      val featsBatch = new Array[Array[IntFloatVector]](indptrBatch.length)
      for (i <- indptrBatch.indices) {
        neighborsBatch(i) = neighbors.slice(splitStart, indptrBatch(i))
        featsBatch(i) = features.slice(splitStart, indptrBatch(i))
        splitStart = indptrBatch(i)
      }
      initEdgeFeaturesByBatch(keys.slice(start, end), neighborsBatch, featsBatch)
      start += step
    }
  }

  def initEdgeFeaturesByBatch(batchKeys: Array[Long],
                              batchNeighbors: Array[Array[Long]],
                              batchFeats: Array[Array[IntFloatVector]]): Unit = {
    val features = new Array[IElement](batchFeats.length)
    for (i <- features.indices) {
      features(i) = new LongEdgeFeats(batchNeighbors(i), batchFeats(i))
    }

    val func = new InitEdgeFeats(new GeneralInitParam(graph.id, batchKeys, features))
    graph.asyncPsfUpdate(func).get()
  }

  def initNodeFeatures(keys: Array[Long], features: Array[IntFloatVector],
                       numBatch: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initNodeFeaturesByBatch(keys.slice(start, end), features.slice(start, end))
      start += step
    }
  }

  def initNodeFeaturesByBatch(batchKeys: Array[Long], batchFeatures: Array[IntFloatVector]): Unit = {
    val features = new Array[IElement](batchFeatures.length)
    for (i <- features.indices) {
      features(i) = new Feature(batchFeatures(i))
    }

    val func = new InitNodeFeats(new GeneralInitParam(graph.id, batchKeys, features))
    graph.asyncPsfUpdate(func).get()
    println(s"init ${batchKeys.length} node features")
  }

  def getFeatures(keys: Array[Long]): Long2ObjectOpenHashMap[IntFloatVector] = {
    graph.psfGet(new GetNodeFeats(new GetNodeAttrsParam(graph.id, keys)))
      .asInstanceOf[GetNodeFeatsResult].getnodeIdToFeats
  }

  def sampleFeatures(size: Int): Array[IntFloatVector] = {
    val features = new Array[IntFloatVector](size)
    var start = 0
    var nxtSize = size
    while (start < size) {
      val func = new SampleNodeFeat(new SampleNodeFeatParam(graph.id, nxtSize))
      val res = graph.psfGet(func).asInstanceOf[SampleNodeFeatResult].getNodeFeats
      Array.copy(res, 0, features, start, math.min(res.length, size - start))
      start += res.length
      nxtSize = (nxtSize + 1) / 2
    }
    features
  }

  def sampleNeighborsWithType(keys: Array[Long],
                              count: Int = -1,
                              sampleType: SampleType = SampleType.NODE): (Long2ObjectOpenHashMap[Array[Long]],
    Long2ObjectOpenHashMap[Array[Int]], Long2ObjectOpenHashMap[Array[Int]]) = {
    if(sampleType == SampleType.SIMPLE) {
      throw new InvalidParameterException("Sample with type only support type: NODE, EDGE and NODE_AND_EDGE");
    }

    val result = graph.psfGet(
      new SampleNeighborWithType(new SampleNeighborWithTypeParam(graph.id, keys, count, sampleType)))
      .asInstanceOf[SampleNeighborResultWithType]
    (result.getNodeIdToSampleNeighbors, result.getNodeIdToSampleNeighborsType, result.getNodeIdToSampleEdgeType)
  }

  def sampleNeighbors(keys: Array[Long], count: Int): Long2ObjectOpenHashMap[Array[Long]] = {
    graph.psfGet(new SampleNeighbor(new SampleNeighborParam(graph.id, keys, count)))
      .asInstanceOf[SampleNeighborResult].getNodeIdToSampleNeighbors
  }

  def sampleEdgeFeatures(keys: Array[Long], count: Int): (Long2ObjectOpenHashMap[Array[Long]], Long2ObjectOpenHashMap[Array[IntFloatVector]])= {
    val param = new SampleEdgeFeatParam(graph.id, keys, count)
    val func = new SampleEdgeFeat(param)
    val result = graph.psfGet(func).asInstanceOf[SampleEdgeFeatResult]
    (result.getNodeIdToSampleNeighbors, result.getNodeIdToSampleEdgeFeats)
  }

  def getNodesWithOutDegree(index: Int, numPartitions: Int): Array[Long] = {
    val pkeys = PSAgentContext.get().getMatrixMetaManager.getPartitions(graph.id)
    val partIds = new IntArrayList()
    val it = pkeys.iterator()
    while (it.hasNext) {
      partIds.add(it.next().getPartitionId)
    }

    val sortedIds = partIds.toIntArray()
    IntArrays.quickSort(sortedIds)

    val myPartIds = new IntArrayList()
    for (id <- sortedIds)
      if (id % numPartitions == index)
        myPartIds.add(id)

    if (myPartIds.size() > 0) {
      val func = new GetNodes(new GetNodesParam(graph.id, myPartIds.toIntArray))
      graph.psfGet(func).asInstanceOf[GetRowResult].getRow
        .asInstanceOf[IntLongVector].getStorage.getValues
    } else
      new Array[Long](0)
  }

  def saveFeatEmbed(featEmbedPath: String): Unit = ???

  /**
    * Dump the matrices on PS to HDFS
    *
    * @param checkpointId checkpoint id
    */
  def checkpointMatrices(checkpointId: Int): Unit = {
    val matrixNames = new Array[String](4)
    matrixNames(0) = "graph"
    matrixNames(1) = "weights"
    matrixNames(2) = "labels"
    matrixNames(3) = "testLabels"
    CheckpointUtils.checkpoint(checkpointId, matrixNames)
  }

}

private[gcn]
object GNNPSModel {
  def apply(minId: Long, maxId: Long, weightSize: Int, optim: AsyncOptim,
            index: RDD[Long], psNumPartition: Int,
            useBalancePartition: Boolean = false): GNNPSModel = {
    val numNode = index.distinct().count()
    // create graph matrix context
    val graphModelContext = new ModelContext(psNumPartition, minId, maxId, numNode, "graph",
      SparkContext.getOrCreate().hadoopConfiguration)
    val graphMatrixContext = ModelContextUtils.createMatrixContext(graphModelContext,
      RowType.T_ANY_LONGKEY_SPARSE, classOf[GraphNode])

    // create labels matrix context
    val labelsModelContext = new ModelContext(psNumPartition, minId, maxId, numNode, "labels",
      SparkContext.getOrCreate().hadoopConfiguration)
    val labelsMatrixContext = ModelContextUtils.createMatrixContext(labelsModelContext, RowType.T_FLOAT_SPARSE_LONGKEY)

    // create testLabels matrix context
    val testLabelsModelContext = new ModelContext(psNumPartition, minId, maxId, numNode, "testLabels",
      SparkContext.getOrCreate().hadoopConfiguration)
    val testLabelsMatrixContext = ModelContextUtils.createMatrixContext(testLabelsModelContext,
      RowType.T_FLOAT_SPARSE_LONGKEY)

    if (!graphModelContext.isUseHashPartition && useBalancePartition)
      LoadBalancePartitioner.partition(index, psNumPartition, graphMatrixContext)

    // create weights matrix context
    val maxColBlock = if (weightSize > psNumPartition) weightSize / psNumPartition else 10
    val weightsMatrixContext = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weightsMatrixContext.setRowType(RowType.T_FLOAT_DENSE)
    weightsMatrixContext.setPartitionerClass(classOf[ColumnRangePartitioner])
    weightsMatrixContext.setMaxColNumInBlock(maxColBlock)

    // create matrixs
    val graphMatrix = PSMatrix.matrix(graphMatrixContext)
    val labels = PSVector.vector(labelsMatrixContext)
    val testLabels = PSVector.vector(testLabelsMatrixContext)
    val weights = PSVector.vector(weightsMatrixContext)

    new GNNPSModel(graphMatrix, weights, labels, testLabels)
  }
}
