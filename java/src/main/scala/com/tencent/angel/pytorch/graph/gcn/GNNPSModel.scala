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

import java.util.{ArrayList => JArrayList}

import com.tencent.angel.graph.client.getnodefeats2.{GetNodeFeats, GetNodeFeatsParam, GetNodeFeatsResult}
import com.tencent.angel.graph.client.getnodes.{GetNodes, GetNodesParam}
import com.tencent.angel.graph.client.initNeighbor5.{InitNeighbor => InitNeighbor5, InitNeighborParam => InitNeighborParam5}
import com.tencent.angel.graph.client.initnodefeats4.{InitNodeFeats => InitNodeFeats4, InitNodeFeatsParam => InitNodeFeatsParam4}
import com.tencent.angel.graph.client.sampleFeats.{SampleNodeFeats, SampleNodeFeatsParam, SampleNodeFeatsResult}
import com.tencent.angel.graph.client.sampleneighbor3.{SampleNeighbor => SampleNeighbor3, SampleNeighborParam => SampleNeighborParam3, SampleNeighborResult => SampleNeighborResult3}
import com.tencent.angel.graph.client.sampleneighbor4.{SampleNeighbor => SampleNeighbor4, SampleNeighborParam => SampleNeighborParam4}
import com.tencent.angel.graph.client.summary.{NnzEdge, NnzFeature, NnzNeighbor, NnzNode}
import com.tencent.angel.graph.data.Node
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.vector.{IntFloatVector, IntLongVector, LongFloatVector}
import com.tencent.angel.ml.matrix.psf.aggr.enhance.ScalarAggrResult
import com.tencent.angel.ml.matrix.psf.get.getrow.GetRowResult
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.partition.LoadBalancePartitioner
import com.tencent.angel.graph.psf.gcn.{GetLabels, GetLabelsResult}
import com.tencent.angel.spark.models.impl.{PSMatrixImpl, PSVectorImpl}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import com.tencent.angel.spark.util.VectorUtils
import it.unimi.dsi.fastutil.ints.{IntArrayList, IntArrays}
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, Long2ObjectOpenHashMap, LongArrayList}
import org.apache.spark.rdd.RDD
import com.tencent.angel.pytorch.utils.CheckpointUtils

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
    val func = new GetLabels(labels.poolId, keys.clone())
    labels.psfGet(func).asInstanceOf[GetLabelsResult].getVector
  }

  def readTestLabels(keys: Array[Long]): LongFloatVector = {
    val func = new GetLabels(testLabels.poolId, keys.clone())
    testLabels.psfGet(func).asInstanceOf[GetLabelsResult].getVector
  }

  def setLabels(value: LongFloatVector): Unit =
    labels.update(value)

  def setTestLabels(value: LongFloatVector): Unit =
    testLabels.update(value)

  def nnzLabels(): Long =
    VectorUtils.size(labels)

  def nnzTestLabels(): Long =
    VectorUtils.size(testLabels)

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

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    numBatch: Int): Unit = {
    val step = keys.length / numBatch
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initNeighbors(keys, indptr, neighbors, start, end)
      start += step
    }
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    numBatch: Int): Unit = {
    val step = keys.length / numBatch
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initNeighbors(keys, indptr, neighbors, types, start, end)
      start += step
    }
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    start: Int,
                    end: Int): Unit = {
    val param = new InitNeighborParam5(graph.id, keys, indptr, neighbors, start, end)
    val func = new InitNeighbor5(param)
    graph.psfUpdate(func).get()
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    start: Int,
                    end: Int): Unit = {
    val param = new InitNeighborParam5(graph.id, keys, indptr, neighbors, types, start, end)
    val func = new InitNeighbor5(param)
    graph.psfUpdate(func).get()
  }

  def initNodeFeatures(keys: Array[Long], features: Array[IntFloatVector],
                       numBatch: Int): Unit = {
    val step = keys.length / numBatch
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initNodeFeatures(keys, features, start, end)
      start += step
    }
  }

  def initNodeFeatures(keys: Array[Long], features: Array[IntFloatVector],
                       start: Int, end: Int): Unit = {
    val param = new InitNodeFeatsParam4(graph.id, keys, features, start, end)
    val func = new InitNodeFeats4(param)
    graph.psfUpdate(func).get()
  }

  def getFeatures(keys: Array[Long]): Long2ObjectOpenHashMap[IntFloatVector] = {
    val func = new GetNodeFeats(new GetNodeFeatsParam(graph.id, keys.clone()))
    graph.psfGet(func).asInstanceOf[GetNodeFeatsResult].getResult
  }

  def sampleFeatures(size: Int): Array[IntFloatVector] = {
    val features = new Array[IntFloatVector](size)
    var start = 0
    var nxtSize = size
    while (start < size) {
      val func = new SampleNodeFeats(new SampleNodeFeatsParam(graph.id, nxtSize))
      val res = graph.psfGet(func).asInstanceOf[SampleNodeFeatsResult].getResult
      Array.copy(res, 0, features, start, math.min(res.length, size - start))
      start += res.length
      nxtSize = (nxtSize + 1) / 2
    }
    features
  }

  def sampleNeighbors(keys: Array[Long], count: Int): Long2ObjectOpenHashMap[Array[Long]] = {
    val func = new SampleNeighbor3(new SampleNeighborParam3(graph.id, keys.clone(), count))
    graph.psfGet(func).asInstanceOf[SampleNeighborResult3].getNodeIdToNeighbors
  }

  def sampleNeighbors(keys: Array[Long], count: Int,
                      index: Long2IntOpenHashMap,
                      srcs: LongArrayList,
                      dsts: LongArrayList,
                      types: LongArrayList): Unit = {
    val param = new SampleNeighborParam4(graph.id, keys, count, types != null)
    val func = new SampleNeighbor4(param, index, srcs, dsts, types)
    graph.psfGet(func)
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

  /**
    * Dump the matrices on PS to HDFS
    *
    * @param checkpointId checkpoint id
    */
  def checkpointMatrices(checkpointId: Int) = {
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
    val graph = new MatrixContext("graph", 1, minId, maxId)
    graph.setRowType(RowType.T_ANY_LONGKEY_SPARSE)
    graph.setValueType(classOf[Node])

    val labels = new MatrixContext("labels", 1, minId, maxId)
    labels.setRowType(RowType.T_FLOAT_SPARSE_LONGKEY)

    val testLabels = new MatrixContext("testLabels", 1, minId, maxId)
    testLabels.setRowType(RowType.T_FLOAT_SPARSE_LONGKEY)

    if (useBalancePartition)
      LoadBalancePartitioner.partition(index, maxId, psNumPartition, graph, 0.5F)

    val weights = new MatrixContext("weights", optim.getNumSlots, weightSize)
    weights.setRowType(RowType.T_FLOAT_DENSE)
    weights.setPartitionerClass(classOf[ColumnRangePartitioner])

    val list = new JArrayList[MatrixContext]()
    list.add(graph)
    list.add(weights)
    list.add(labels)
    list.add(testLabels)

    PSAgentContext.get().getMasterClient.createMatrices(list, 10000L)
    val graphId = PSAgentContext.get().getMasterClient.getMatrix("graph").getId
    val weightsId = PSAgentContext.get().getMasterClient.getMatrix("weights").getId
    val labelsId = PSAgentContext.get().getMasterClient.getMatrix("labels").getId
    val testLabelsId = PSAgentContext.get().getMasterClient.getMatrix("testLabels").getId

    new GNNPSModel(new PSMatrixImpl(graphId, 1, maxId, graph.getRowType),
      new PSVectorImpl(weightsId, 0, weights.getColNum, weights.getRowType),
      new PSVectorImpl(labelsId, 0, maxId, labels.getRowType),
      new PSVectorImpl(testLabelsId, 0, maxId, testLabels.getRowType))
  }
}
