package com.tencent.angel.pytorch.graph.gcn

import com.carrotsearch.hppc.LongArrayList
import com.tencent.angel.exception.InvalidParameterException
import com.tencent.angel.graph.client.psf.get.utils.GetNodeAttrsParam
import com.tencent.angel.graph.client.psf.init.GeneralInitParam
import com.tencent.angel.graph.client.psf.init.inittypeneighbors.InitTypeNeighbors
import com.tencent.angel.graph.client.psf.sample.sampleneighbor._
import com.tencent.angel.graph.common.param.ModelContext
import com.tencent.angel.graph.data._
import com.tencent.angel.graph.utils.ModelContextUtils
import com.tencent.angel.ml.math2.vector.{IntFloatVector, IntLongVector, LongIntVector, Vector}
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.ps.storage.vector.element.IElement
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.graph.client.psf.universalembedding._
import com.tencent.angel.graph.client.psf.get.utils.GetFloatArrayAttrsResult
import com.tencent.angel.model.{MatrixLoadContext, MatrixSaveContext, ModelLoadContext, ModelSaveContext}
import com.tencent.angel.pytorch.init.InitUtils
import com.tencent.angel.pytorch.utils.CheckpointUtils
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.util.LogUtils
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap
import org.apache.spark.SparkContext
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import org.apache.spark.rdd.RDD
import java.util.concurrent.TimeUnit
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

private[gcn]
class EmbeddingGNNPSModel(graph: PSMatrix,
                          weights: PSVector,
                          labels: PSVector,
                          testLabels: PSVector,
                          nodeTypes: PSVector,
                          context: PSMatrix,
                          index2Node: PSVector,
                          embeddings: Map[Int, PSMatrix],
                          embedDims: Map[Int, Int]) extends GNNPSModel(graph, weights, labels, testLabels) {

  def readNodeTypes(keys: Array[Long]): LongIntVector =
    nodeTypes.pull(keys.clone()).asInstanceOf[LongIntVector]

  def setNodeTypes(value: LongIntVector): Unit =
    nodeTypes.update(value)

  def readIndex2Node(keys: Array[Int]): IntLongVector =
    index2Node.pull(keys.clone()).asInstanceOf[IntLongVector]

  def setIndex2Node(value: IntLongVector): Unit =
    index2Node.update(value)

  def initContext(e: IndexedSeq[Long],
                  embeddingDim: Int,
                  initMethod: String,
                  numNodes: Long,
                  optim: AsyncOptim,
                  seed: Int,
                  mean: Float=0.0f,
                  std: Float=1.0f): Unit = {
    val fin = numNodes
    val fout = embeddingDim.toLong
    val initFunc = InitUtils.apply(initMethod, fin, fout, mean=mean, std=std)

    context.asyncPsfUpdate(new UniversalEmbeddingInitAsNodes(
      new UniversalEmbeddingInitParam(context.id, e.toArray, seed, embeddingDim, optim.getNumSlots(), initFunc.getFloats(), initFunc.getInts())))
      .get(120000, TimeUnit.MILLISECONDS)
  }

  def initExtraContext(keys: Array[Long], features: Array[IntFloatVector], embeddingDim: Int, numBatch: Int, slot: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initExtraContextByBatch(keys.slice(start, end), features.slice(start, end), embeddingDim, slot)
      start += step
    }
  }

  def initExtraContextByBatch(batchKeys: Array[Long], batchEmbeddings: Array[IntFloatVector], embeddingDim: Int, slot: Int): Unit = {
    val features = new Array[IElement](batchEmbeddings.length)
    for (i <- features.indices) {
      features(i) = new EmbeddingOrGrad(batchEmbeddings(i).getStorage.getValues)
    }

    val func = new UniversalEmbeddingExtraInitAsNodes(new UniversalEmbeddingExtraInitParam(context.id, batchKeys, features, embeddingDim, slot))
    context.asyncPsfUpdate(func).get()
  }

  def getContext(dstNodes: Array[Long], negativeSamples: Array[Array[Long]]): Long2ObjectOpenHashMap[Array[Float]] = {
    val all_nodes = dstNodes.union(negativeSamples.flatMap(x => x))
    val func = new UniversalEmbeddingGet(new GetNodeAttrsParam(context.id, all_nodes))
    context.psfGet(func).asInstanceOf[GetFloatArrayAttrsResult].getNodeIdToContents
  }

  def initNeighbors(keys: Array[(Long, Iterable[(Long, Int)])], numBatch: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    var splitStart = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      val batchKeys = new LongArrayList()
      val batchNeighbors = new ArrayBuffer[Int2ObjectOpenHashMap[Array[Long]]]()
      while (splitStart < end) {
        val entry = keys(splitStart)
        val (node, ns) = (entry._1, entry._2.toArray)
        val neighbors = new Int2ObjectOpenHashMap[Array[Long]]
        ns.groupBy(f => f._2).foreach(r => neighbors.put(r._1, r._2.map(_._1)))
        batchKeys.add(node)
        batchNeighbors.append(neighbors)
        splitStart += 1
      }
      initNeighborsByBatch(batchKeys.toArray, batchNeighbors.toArray)
      start += step
    }
  }

  def initNeighborsByBatch(batchKeys: Array[Long], batchNeighbors: Array[Int2ObjectOpenHashMap[Array[Long]]]): Unit = {
    val neighbors = new Array[IElement](batchKeys.length)
    for (i <- batchNeighbors.indices) {
      neighbors(i) = new TypeNeighbors(batchNeighbors(i))
    }
    val func = new InitTypeNeighbors(new GeneralInitParam(graph.id, batchKeys.clone(), neighbors))
    graph.asyncPsfUpdate(func).get()
  }

  def embeddingStep(nodeId: Array[Long], grads: Array[IElement], optim: AsyncOptim): Unit = {
    optim.asyncUpdate(context, nodeId, grads).get()
  }

  def sampleNeighborsByType(keys: Array[Long],
                            count: Int = -1,
                            sampleType: SampleType = SampleType.NODE,
                            node_or_edge_type: Int = -1,
                            sampleMethod: SampleMethod = SampleMethod.RANDOM): Long2ObjectOpenHashMap[Neighbor] = {
    if(sampleType == SampleType.SIMPLE) {
      throw new InvalidParameterException("Sample with type only support type: NODE, EDGE and NODE_AND_EDGE");
    }

    val result = graph.psfGet(
      new SampleNeighborByType(new SampleNeighborByTypeParam(graph.id, keys, count, sampleType, node_or_edge_type, sampleMethod, NeighborStorageType.LONGARRAY)))
      .asInstanceOf[SampleNeighborResultByType]
    result.getNodeIdToSampleNeighbors
  }

  override
  def checkpointMatrices(checkpointId: Int): Unit = {
    val matrixNames = new ArrayBuffer[String]()
    matrixNames.append("graph")
    matrixNames.append("weights")
    matrixNames.append("nodeTypes")
    matrixNames.append("context")
    matrixNames.append("index2Node")
    if (embeddings != null) {
      embeddings.keys.foreach(name => matrixNames.append(name+"Embedding"))
    }
    CheckpointUtils.checkpoint(checkpointId, matrixNames.toArray)
  }

  def initEmbeddings(featureIds: Map[Int, RDD[Long]], batchSize: Int, slots: Int, initMethod: String): Unit = {
    val beforeRandomize = System.currentTimeMillis()
    featureIds.foreach{ case (name, ids) =>
      val embeddingDim = embedDims.getOrElse(name, -1)
      val featureNum = ids.count()

      ids.foreachPartition{ part =>
        part.sliding(batchSize, batchSize).foreach{ batchIter =>
          val seed = System.currentTimeMillis().toInt
          initEmbeddingsByBatch(name, batchIter.toArray, seed, slots, featureNum, embeddingDim, initMethod)
        }
      }
    }
    LogUtils.logTime(s"feature embedding successfully Randomized, cost ${(System.currentTimeMillis() - beforeRandomize) / 1000.0}s")
  }

  def initEmbeddingsByBatch(embeddingName: Int, nodeIds: Array[Long], seed: Int, numSlots: Int, featureNum: Long, embeddingDim: Int, initMethod: String): Unit = {
    val fin = featureNum
    val fout = embeddingDim.toLong
    val initFunc = InitUtils.apply(initMethod, fin, fout)
    val embedding = embeddings.getOrElse(embeddingName, null)
    val func = new UniversalEmbeddingInitAsNodes(
      new UniversalEmbeddingInitParam(embedding.id, nodeIds, seed, embeddingDim, numSlots, initFunc.getFloats(), initFunc.getInts()))
    embedding.psfUpdate(func).get()
  }

  def initExtraEmbeddings(keys: Array[Long], features: Array[IntFloatVector],
                          embeddingName: Int, numBatch: Int, slot: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initExtraEmbeddingsByBatch(keys.slice(start, end), features.slice(start, end), embeddingName, slot)
      start += step
    }
  }

  def initExtraEmbeddingsByBatch(batchKeys: Array[Long], batchEmbeddings: Array[IntFloatVector], embeddingName: Int, slot: Int): Unit = {
    val embedding = embeddings.getOrElse(embeddingName, null.asInstanceOf[PSMatrix])
    val dim = embedDims.getOrElse(embeddingName, -1)
    val features = new Array[IElement](batchEmbeddings.length)
    for (i <- features.indices) {
      features(i) = new EmbeddingOrGrad(batchEmbeddings(i).getStorage.getValues)
    }

    val func = new UniversalEmbeddingExtraInitAsNodes(new UniversalEmbeddingExtraInitParam(embedding.id, batchKeys, features, dim, slot))
    embedding.asyncPsfUpdate(func).get()
  }

  def getEmbedding(indices: Array[Long], graphType: Int): Long2ObjectOpenHashMap[Array[Float]] = {
    val embedding = embeddings.getOrElse(graphType, null)
    val func = new UniversalEmbeddingGet(new GetNodeAttrsParam(embedding.id, indices))
    embedding.psfGet(func).asInstanceOf[GetFloatArrayAttrsResult].getNodeIdToContents
  }

  def updateEmbedding(grads: Array[Vector], optim: AsyncOptim, graphType: Int): Unit = {
    val dim = embedDims.getOrElse(graphType, -1)
    optim.update(embeddings.getOrElse(graphType, null.asInstanceOf[PSMatrix]), dim, (0 until dim).toArray, grads)
  }

  def updateEmbedding(grads: Long2ObjectOpenHashMap[Array[Float]], optim: AsyncOptim, graphType: Int): Unit = {
    val embedding = embeddings.getOrElse(graphType, null)
    val gradsElements = new Array[IElement](grads.size())
    val ids = new Array[Long](gradsElements.length)

    val iter = grads.long2ObjectEntrySet().fastIterator()
    var idx = 0
    while (iter.hasNext) {
      val entry = iter.next()
      ids(idx) = entry.getLongKey
      gradsElements(idx) = new EmbeddingOrGrad(entry.getValue)

      idx += 1
    }

    optim.update(embedding, ids, gradsElements)
  }

  override def saveFeatEmbed(featEmbedPath: String): Unit = {
    val ctx = new ModelSaveContext(featEmbedPath)
    val format = classOf[TextUniversalEmbModelOutEmbOutputFormat].getCanonicalName
    embeddings.foreach{ case (name, _) =>
      val embeddingCtx = new MatrixSaveContext(name + "Embedding", format)
      ctx.addMatrix(embeddingCtx)
    }

    PSContext.instance().save(ctx)
    println(s"save user (and item) feature embeddings(in the form of angel model) to $featEmbedPath.")
  }

  def saveContext(contextPath: String): Unit = {
    val ctx = new ModelSaveContext(contextPath)
    val format = classOf[TextUniversalEmbModelOutEmbOutputFormat].getCanonicalName

    val embeddingCtx = new MatrixSaveContext("context", format)
    ctx.addMatrix(embeddingCtx)

    PSContext.instance().save(ctx)
    println(s"save the node context embeddings (in the form of angel model) to $contextPath.")
  }

  override def loadFeatEmbed(featEmbedPath: String): Unit = {
    val ctx = new ModelLoadContext(featEmbedPath)
    embeddings.foreach{ case (name, _) =>
      val embeddingCtx = new MatrixLoadContext(name + "Embedding")
      ctx.addMatrix(embeddingCtx)
    }
    PSContext.getOrCreate(SparkContext.getOrCreate()).load(ctx)
  }

}

private[gcn]
object EmbeddingGNNPSModel {
  def apply(minId: Long,
            maxId: Long,
            weightSize: Int,
            optim: AsyncOptim,
            numNode: Long,
            featureIds: Option[Map[Int, RDD[Long]]],
            embedDims: Map[Int, Int],
            featureDims: Map[Int, Int],
            featureSplitIdxs: Map[Int, Int],
            psNumPartition: Int,
            useBalancePartition: Boolean = false): EmbeddingGNNPSModel = {
    // create graph matrix context
    val graphModelContext = new ModelContext(psNumPartition, minId, maxId, numNode, "graph",
      SparkContext.getOrCreate().hadoopConfiguration)
    val graphMatrixContext = ModelContextUtils.createMatrixContext(graphModelContext,
      RowType.T_ANY_LONGKEY_SPARSE, classOf[GraphNode])

    // Create one ps matrix to hold the output vectors for all node
    val contextModelContext = new ModelContext(psNumPartition, minId, maxId, numNode, "context",
      SparkContext.getOrCreate().hadoopConfiguration)
    val contextMatrixContext = ModelContextUtils.createMatrixContext(contextModelContext,
      RowType.T_ANY_LONGKEY_SPARSE, classOf[UniversalEmbeddingNode])

    val nodeTypeModelContext = new ModelContext(psNumPartition, minId, maxId, numNode, "nodeTypes",
      SparkContext.getOrCreate().hadoopConfiguration)
    val nodeTypeMatrixContext = ModelContextUtils.createMatrixContext(nodeTypeModelContext, RowType.T_INT_SPARSE_LONGKEY)

    val index2NodeModelContext = new ModelContext(psNumPartition, minId, maxId, numNode, "index2Node",
      SparkContext.getOrCreate().hadoopConfiguration)
    val index2NodeMatrixContext = ModelContextUtils.createMatrixContext(index2NodeModelContext, RowType.T_LONG_SPARSE)

    // create weights matrix context
    val maxColBlock = if (weightSize > psNumPartition) weightSize / psNumPartition else 10
    val weightsMatrixContext = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weightsMatrixContext.setRowType(RowType.T_FLOAT_DENSE)
    weightsMatrixContext.setPartitionerClass(classOf[ColumnRangePartitioner])
    weightsMatrixContext.setMaxColNumInBlock(maxColBlock)

    val embeddings = if (embedDims.size > 1) {
      val featureNumMap = new mutable.HashMap[Int, Long]()
      if (featureIds.nonEmpty) {
        featureIds.get.map { case (name, rdd) =>
          var featureNum = rdd.count()
          val num = featureNumMap.getOrElse(name, -1L)
          featureNum = if (featureNum < num) num else featureNum
          featureNumMap.put(name, featureNum)
        }
      }

      featureDims.map{ case(name, _) =>
        val num = featureNumMap.getOrElse(name, -1L)
        val modelContext = new ModelContext(psNumPartition, 0, featureDims.getOrElse(name, 0) + 1L - featureSplitIdxs.getOrElse(name, 0),
          num, name + "Embedding", SparkContext.getOrCreate().hadoopConfiguration)
        val matrixContext = ModelContextUtils.createMatrixContext(modelContext,
          RowType.T_ANY_LONGKEY_SPARSE, classOf[UniversalEmbeddingNode])

        (name, PSMatrix.matrix(matrixContext))
      }.toMap
    } else null

    // create matrixs
    val graphMatrix = PSMatrix.matrix(graphMatrixContext)
    val contextMatrix = PSMatrix.matrix(contextMatrixContext)
    val weights = PSVector.vector(weightsMatrixContext)
    val nodeTypes = PSVector.vector(nodeTypeMatrixContext)
    val index2Node = PSVector.vector(index2NodeMatrixContext)

    new EmbeddingGNNPSModel(graphMatrix, weights, null, null, nodeTypes, contextMatrix, index2Node, embeddings, embedDims)
  }
}
