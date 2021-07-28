package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.exception.InvalidParameterException
import com.tencent.angel.graph.client.psf.get.getnodefeats.{GetNodeFeats, GetNodeFeatsResult}
import com.tencent.angel.graph.client.psf.get.utils.GetNodeAttrsParam
import com.tencent.angel.graph.client.psf.init.GeneralInitParam
import com.tencent.angel.graph.client.psf.init.initedgetypes.InitEdgeTypes
import com.tencent.angel.graph.client.psf.init.initneighbors.InitNeighbor
import com.tencent.angel.graph.client.psf.init.initnodefeats.InitNodeFeats
import com.tencent.angel.graph.client.psf.init.initnodetypes.InitNodeTypes
import com.tencent.angel.graph.client.psf.sample.samplenodefeats.{SampleNodeFeat, SampleNodeFeatParam, SampleNodeFeatResult}
import com.tencent.angel.graph.client.psf.sample.sampleneighbor._
import com.tencent.angel.graph.client.psf.summary.{NnzEdge, NnzFeature, NnzNeighbor, NnzNode}
import com.tencent.angel.graph.common.param.ModelContext
import com.tencent.angel.graph.data._
import com.tencent.angel.graph.utils.ModelContextUtils
import com.tencent.angel.ml.math2.vector.IntFloatVector
import com.tencent.angel.ml.matrix.psf.aggr.enhance.ScalarAggrResult
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.ps.storage.vector.element.IElement
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.pytorch.utils.CheckpointUtils
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class BiSAGEPSModel(userGraph: PSMatrix,
                    itemGraph: PSMatrix,
                    weights: PSVector,
                    labels: PSVector,
                    testLabels: PSVector) extends
  GNNPSModel(userGraph, weights, labels, testLabels) {

  /* summary functions */
  def nnzFeatures(graphType: Int): Long = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    val func = new NnzFeature(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }
  def nnzNodes(graphType: Int): Long = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    val func = new NnzNode(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }
  def nnzNeighbors(graphType: Int): Long = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    val func = new NnzNeighbor(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }
  def nnzEdge(graphType: Int): Long = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    val func = new NnzEdge(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    edgeTypes: Array[Int],
                    dstTypes: Array[Int],
                    graphType: Int,
                    numBatch: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    var splitStart = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      val indptrBatch = indptr.slice(start + 1, end + 1)
      val neighborsBatch = new Array[Array[Long]](indptrBatch.length)
      val nodeTypesBatch = new Array[Array[Int]](indptrBatch.length)
      val edgeTypesBatch = new Array[Array[Int]](indptrBatch.length)
      for (i <- indptrBatch.indices) {
        neighborsBatch(i) = neighbors.slice(splitStart, indptrBatch(i))
        nodeTypesBatch(i) = dstTypes.slice(splitStart, indptrBatch(i))
        edgeTypesBatch(i) = edgeTypes.slice(splitStart, indptrBatch(i))
        splitStart = indptrBatch(i)
      }

      initNeighborsByBatch(keys.slice(start, end), neighborsBatch, nodeTypesBatch, edgeTypesBatch, graphType)
      start += step
    }
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    graphType: Int,
                    numBatch: Int,
                    hasNodeType: Boolean): Unit = {
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
      initNeighborsByBatch(keys.slice(start, end), neighborsBatch, typesBatch, graphType, hasNodeType)
      start += step
    }
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    graphType: Int,
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
      initNeighborsByBatch(keys.slice(start, end), neighborsBatch, graphType)
      start += step
    }
  }

  def initNeighborsByBatch(batchKeys: Array[Long],
                           batchNeighbors: Array[Array[Long]],
                           graphType: Int): Unit = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    val neighbors = new Array[IElement](batchKeys.length)
    for (i <- batchNeighbors.indices) {
      neighbors(i) = new LongNeighbor(batchNeighbors(i))
    }
    val func = new InitNeighbor(new GeneralInitParam(graph.id, batchKeys, neighbors))
    graph.asyncPsfUpdate(func).get()
    println(s"init ${batchKeys.length} neighbors")
  }

  def initNeighborsByBatch(batchKeys: Array[Long],
                           batchNeighbors: Array[Array[Long]],
                           batchTypes: Array[Array[Int]],
                           graphType: Int,
                           hasNodeType: Boolean): Unit = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    val neighbors = new Array[IElement](batchKeys.length)
    for (i <- batchNeighbors.indices) {
      neighbors(i) = new LongNeighbor(batchNeighbors(i))
    }
    val func = new InitNeighbor(new GeneralInitParam(graph.id, batchKeys.clone(), neighbors))
    graph.asyncPsfUpdate(func).get()
    println(s"init ${batchKeys.length} neighbors")
    if (hasNodeType) {
      val nodeTypes = new Array[IElement](batchKeys.length)
      for (i <- batchTypes.indices) {
        nodeTypes(i) = new NodeType(batchTypes(i))
      }
      val nodeFunc = new InitNodeTypes(new GeneralInitParam(graph.id, batchKeys.clone(), nodeTypes))
      graph.asyncPsfUpdate(nodeFunc).get()
      println(s"init ${batchKeys.length} node types")
    } else {
      val edgeTypes = new Array[IElement](batchKeys.length)
      for (i <- batchTypes.indices) {
        edgeTypes(i) = new EdgeType(batchTypes(i))
      }
      val edgeFunc = new InitEdgeTypes(new GeneralInitParam(graph.id, batchKeys.clone(), edgeTypes))
      graph.asyncPsfUpdate(edgeFunc).get()
      println(s"init ${batchKeys.length} edge types")
    }
  }

  def initNeighborsByBatch(batchKeys: Array[Long],
                           batchNeighbors: Array[Array[Long]],
                           batchNodeTypes: Array[Array[Int]],
                           batchEdgeTypes: Array[Array[Int]],
                           graphType: Int): Unit = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    val neighbors = new Array[IElement](batchKeys.length)
    for (i <- batchNeighbors.indices) {
      neighbors(i) = new LongNeighbor(batchNeighbors(i))
    }
    val func = new InitNeighbor(new GeneralInitParam(graph.id, batchKeys.clone(), neighbors))
    graph.asyncPsfUpdate(func).get()
    println(s"init ${batchKeys.length} neighbors")

    val nodeTypes = new Array[IElement](batchKeys.length)
    for (i <- batchNodeTypes.indices) {
      nodeTypes(i) = new NodeType(batchNodeTypes(i))
    }
    val nodeFunc = new InitNodeTypes(new GeneralInitParam(graph.id, batchKeys.clone(), nodeTypes))
    graph.asyncPsfUpdate(nodeFunc).get()
    println(s"init ${batchKeys.length} node types")

    val edgeTypes = new Array[IElement](batchKeys.length)
    for (i <- batchEdgeTypes.indices) {
      edgeTypes(i) = new EdgeType(batchEdgeTypes(i))
    }
    val edgeFunc = new InitEdgeTypes(new GeneralInitParam(graph.id, batchKeys.clone(), edgeTypes))
    graph.asyncPsfUpdate(edgeFunc).get()
    println(s"init ${batchKeys.length} edge types")
  }

  def initNodeFeatures(keys: Array[Long], features: Array[IntFloatVector],
                       graphType: Int, numBatch: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initNodeFeaturesByBatch(keys.slice(start, end), features.slice(start, end), graphType)
      start += step
    }
  }

  def initNodeFeaturesByBatch(batchKeys: Array[Long], batchFeatures: Array[IntFloatVector], graphType: Int): Unit = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    val features = new Array[IElement](batchFeatures.length)
    for (i <- features.indices) {
      features(i) = new Feature(batchFeatures(i))
    }

    val func = new InitNodeFeats(new GeneralInitParam(graph.id, batchKeys, features))
    graph.asyncPsfUpdate(func).get()
    println(s"init ${batchKeys.length} node features")
  }

  def sampleNeighbors(keys: Array[Long], count: Int, graphType: Int): Long2ObjectOpenHashMap[Array[Long]] = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    graph.psfGet(new SampleNeighbor(new SampleNeighborParam(graph.id, keys, count)))
      .asInstanceOf[SampleNeighborResult].getNodeIdToSampleNeighbors
  }

  def sampleNeighborsWithType(keys: Array[Long],
                              count: Int,
                              sampleType: SampleType,
                              graphType: Int): (Long2ObjectOpenHashMap[Array[Long]],
    Long2ObjectOpenHashMap[Array[Int]], Long2ObjectOpenHashMap[Array[Int]]) = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    if(sampleType == SampleType.SIMPLE) {
      throw new InvalidParameterException("Sample with type only support type: NODE, EDGE and NODE_AND_EDGE");
    }
    val result = graph.psfGet(
      new SampleNeighborWithType(new SampleNeighborWithTypeParam(graph.id, keys, count, sampleType)))
      .asInstanceOf[SampleNeighborResultWithType]
    (result.getNodeIdToSampleNeighbors, result.getNodeIdToSampleNeighborsType, result.getNodeIdToSampleEdgeType)
  }

  def sampleNeighborsWithTypeWithFilter(nodeIds: Array[Long], filterWithNeighKeys: Array[Long],
                                        filterWithoutNeighKeys: Array[Long], count: Int = -1,
                                        sampleType: SampleType = SampleType.NODE, graphType: Int = 0): (
    Long2ObjectOpenHashMap[Array[Long]], Long2ObjectOpenHashMap[Array[Int]], Long2ObjectOpenHashMap[Array[Int]]) = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    if(sampleType != SampleType.NODE && sampleType != SampleType.EDGE) {
      throw new InvalidParameterException("Sample with filter only support type: NODE, EDGE");
    }

    val result = graph.psfGet(
      new SampleNeighborWithFilter(
        new SampleNeighborWithFilterParam(
          graph.id, nodeIds, count, sampleType, filterWithNeighKeys, filterWithoutNeighKeys)))
      .asInstanceOf[SampleNeighborResultWithType]
    (result.getNodeIdToSampleNeighbors, result.getNodeIdToSampleNeighborsType, result.getNodeIdToSampleEdgeType)
  }

  def getFeatures(keys: Array[Long], graphType: Int): Long2ObjectOpenHashMap[IntFloatVector] = {
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraph
    graph.psfGet(new GetNodeFeats(new GetNodeAttrsParam(graph.id, keys)))
      .asInstanceOf[GetNodeFeatsResult].getnodeIdToFeats
  }

  def sampleFeatures(size: Int, graphType: Int): Array[IntFloatVector] = {
    val features = new Array[IntFloatVector](size)
    var start = 0
    var nxtSize = size
    val graph = if (graphType == 0) userGraph else itemGraph   // choose userGraph or itemGraphs
    while (start < size) {
      val func = new SampleNodeFeat(new SampleNodeFeatParam(graph.id, nxtSize))
      val res = graph.psfGet(func).asInstanceOf[SampleNodeFeatResult].getNodeFeats
      Array.copy(res, 0, features, start, math.min(res.length, size - start))
      start += res.length
      nxtSize = (nxtSize + 1) / 2
    }
    features
  }

  /**
    * Dump the matrices on PS to HDFS
    *
    * @param checkpointId checkpoint id
    */
  override
  def checkpointMatrices(checkpointId: Int): Unit = {
    val matrixNames = new Array[String](5)
    matrixNames(0) = "userGraph"
    matrixNames(1) = "itemGraph"
    matrixNames(2) = "weights"
    matrixNames(3) = "labels"
    matrixNames(4) = "testLabels"
    CheckpointUtils.checkpoint(checkpointId, matrixNames)
  }
}

private[gcn]
object BiSAGEPSModel {
  def apply(userMinId: Long, userMaxId: Long, itemMinId: Long, itemMaxId: Long, weightSize: Int, optim: AsyncOptim,
            index: RDD[Long], psNumPartition: Int,
            useBalancePartition: Boolean = false): BiSAGEPSModel = {
    val userNumNode = index.distinct().count()
    //create user graph matrix context
    val userModelContext = new ModelContext(psNumPartition, userMinId, userMaxId, userNumNode, "userGraph",
      SparkContext.getOrCreate().hadoopConfiguration)
    val userGraphMatrixContext = ModelContextUtils.createMatrixContext(userModelContext,
      RowType.T_ANY_LONGKEY_SPARSE, classOf[GraphNode])

    // create labels matrix context
    val labelsModelContext = new ModelContext(psNumPartition, userMinId, userMaxId, userNumNode, "labels",
      SparkContext.getOrCreate().hadoopConfiguration)
    val labelsMatrixContext = ModelContextUtils.createMatrixContext(labelsModelContext, RowType.T_FLOAT_SPARSE_LONGKEY)

    // create testLabels matrix context
    val testLabelsModelContext = new ModelContext(psNumPartition, userMinId, userMaxId, userNumNode,
      "testLabels", SparkContext.getOrCreate().hadoopConfiguration)
    val testLabelsMatrixContext = ModelContextUtils.createMatrixContext(testLabelsModelContext, RowType.T_FLOAT_SPARSE_LONGKEY)

    // create item graph matrix context
    val itemModelContext = new ModelContext(psNumPartition, itemMinId, itemMaxId, -1, "itemGraph",
      SparkContext.getOrCreate().hadoopConfiguration)
    val itemGraphMatrixContext = ModelContextUtils.createMatrixContext(itemModelContext,
      RowType.T_ANY_LONGKEY_SPARSE, classOf[GraphNode])

    if (!userModelContext.isUseHashPartition && useBalancePartition)
      LoadBalancePartitioner.partition(index, psNumPartition, userGraphMatrixContext)

    // create weights matrix context
    val weights = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weights.setRowType(RowType.T_FLOAT_DENSE)
    weights.setPartitionerClass(classOf[ColumnRangePartitioner])

    // create matrix
    val userGraph = PSMatrix.matrix(userGraphMatrixContext)
    val itemGraph = PSMatrix.matrix(itemGraphMatrixContext)
    val weightsVec = PSVector.vector(weights)
    val labels = PSVector.vector(labelsMatrixContext)
    val testLabels = PSVector.vector(testLabelsMatrixContext)

    new BiSAGEPSModel(userGraph, itemGraph, weightsVec, labels, testLabels)
  }
}