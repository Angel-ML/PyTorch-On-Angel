package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.graph.client.psf.get.getnodefeats.{GetMultiNodeFeats, GetNodeFeatsResult}
import com.tencent.angel.graph.client.psf.get.utils.{GetFloatArrayAttrsResult, GetNodeAttrsParam}
import com.tencent.angel.graph.client.psf.init.initedgeweights.InitEdgeWeightsByName
import com.tencent.angel.graph.client.psf.init.{GeneralInitByNameParam, GeneralInitParam}
import com.tencent.angel.graph.client.psf.init.initneighbors.{InitAliasTableByName, InitNeighborByName}
import com.tencent.angel.graph.client.psf.init.initnodefeats.InitMultiNodeFeats
import com.tencent.angel.graph.client.psf.sample.sampleneighbor.{SampleMethod, SampleNeighborByName, SampleNeighborByNameParam, SampleNeighborResult}
import com.tencent.angel.graph.client.psf.sample.samplenodefeats.{SampleNodeFeat, SampleNodeFeatParam, SampleNodeFeatResult}
import com.tencent.angel.graph.client.psf.universalembedding.{UniversalEmbeddingExtraInitParam, _}
import com.tencent.angel.graph.common.param.ModelContext
import com.tencent.angel.graph.data._
import com.tencent.angel.graph.utils.ModelContextUtils
import com.tencent.angel.ml.math2.vector.{FloatVector, IntFloatVector, LongFloatVector, Vector}
import com.tencent.angel.ml.matrix.psf.aggr.Size
import com.tencent.angel.ml.matrix.psf.aggr.enhance.ScalarAggrResult
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.model.{MatrixLoadContext, MatrixSaveContext, ModelLoadContext, ModelSaveContext}
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.ps.storage.vector.element.IElement
import com.tencent.angel.pytorch.init.InitUtils
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.utils.CheckpointUtils
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, HashMap}

class HGNNPSModel(graphs: Map[String, PSMatrix],
                  weights: PSVector,
                  labels: Map[String, PSVector],
                  testLabels: Map[String, PSVector],
                  embeddings: Map[String, PSMatrix],
                  embedDims: Map[String, Int],
                  storageTypes: Map[String, NeighborStorageType]) extends
  GNNPSModel(null, weights, null, null) {

  // the default pull method will return keys even those not exists on servers
  def readLabels(name: String, keys: Array[Long]): LongFloatVector = {
    if (labels != null && labels.contains(name) && keys != null) {
      labels.getOrElse(name, null).pull(keys.clone()).asInstanceOf[LongFloatVector]
    } else null
  }

  // this method will not return keys that do not exist on servers
  def readLabels2(name:String, keys: Array[Long]): LongFloatVector = {
    import com.tencent.angel.spark.ml.psf.gcn.{GetLabels, GetLabelsResult}
    if (labels != null && labels.contains(name) && keys != null) {
      val func = new GetLabels(labels.getOrElse(name, null).poolId, keys.clone())
      labels.getOrElse(name, null).psfGet(func).asInstanceOf[GetLabelsResult].getVector
    } else null
  }

  def readTestLabels(name: String, keys: Array[Long]): LongFloatVector = {
    import com.tencent.angel.spark.ml.psf.gcn.{GetLabels, GetLabelsResult}
    if (testLabels != null && testLabels.contains(name) && keys != null) {
      val func = new GetLabels(testLabels.getOrElse(name, null).poolId, keys.clone())
      testLabels.getOrElse(name, null).psfGet(func).asInstanceOf[GetLabelsResult].getVector
    } else null
  }

  def setLabels(name: String, value: LongFloatVector): Unit =
    labels.getOrElse(name, null).update(value)

  def setTestLabels(name: String, value: LongFloatVector): Unit =
    testLabels.getOrElse(name, null).update(value)

  def nnzLabels(name: String): Long = {
    val label = labels.getOrElse(name, null)
    label.psfGet(new Size(label.poolId, label.id)).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def nnzTestLabels(name: String): Long = {
    val testLabel = testLabels.getOrElse(name, null)
    testLabel.psfGet(new Size(testLabel.poolId, testLabel.id)).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def initNeighbors(keys: Array[Long], indptr: Array[Int], neighbors: Array[Long], numBatch: Int, edgeName: String): Unit = {
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
      val nodes = edgeName.split("-")
      initNeighborsByBatch(graphs.getOrElse(nodes(0), null), keys.slice(start, end), neighborsBatch, nodes(1))
      start += step
    }
  }

  def initNeighborsByBatch(graph: PSMatrix, batchKeys: Array[Long], batchNeighbors: Array[Array[Long]], name: String): Unit = {
    val neighbors = new Array[IElement](batchKeys.length)
    for (i <- batchNeighbors.indices) {
      if (storageTypes.getOrElse(name, NeighborStorageType.LONGARRAY) == NeighborStorageType.LONGARRAY) {
        neighbors(i) = new LongNeighbor(batchNeighbors(i))
      } else {
        neighbors(i) = new IntNeighbor(batchNeighbors(i).map(_.toInt))
      }
    }

    val func = new InitNeighborByName(new GeneralInitByNameParam(graph.id, batchKeys, neighbors, name))
    graph.asyncPsfUpdate(func).get()
  }

  def initWeights(keys: Array[Long],
                  indptr: Array[Int],
                  weights: Array[Float],
                  numBatch: Int,
                  edgeName: String): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    var splitStart = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      val indptrBatch = indptr.slice(start + 1, end + 1)
      val weightsBatch = new Array[Array[Float]](indptrBatch.length)
      for (i <- indptrBatch.indices) {
        weightsBatch(i) = weights.slice(splitStart, indptrBatch(i))
        splitStart = indptrBatch(i)
      }
      val nodes = edgeName.split("-")
      initWeightsByBatch(graphs.getOrElse(nodes(0), null), keys.slice(start, end), weightsBatch, nodes(1))
      start += step
    }
  }

  def initWeightsByBatch(graph: PSMatrix, batchKeys: Array[Long], batchWeights: Array[Array[Float]], name: String): Unit = {
    val weights = new Array[IElement](batchKeys.length)
    for (i <- batchWeights.indices) {
      weights(i) = new Weights(batchWeights(i))
    }
    val func = new InitEdgeWeightsByName(new GeneralInitByNameParam(graph.id, batchKeys, weights, name))
    graph.asyncPsfUpdate(func).get()
    println(s"init ${batchKeys.length} edge weights")
  }

  def initAliasTable(neighborTable: Seq[(Long, (Array[Float], Array[Int]))], name: String):
  Unit = {
    val nodeNames = name.split("-")
    val nodeIds = new Array[Long](neighborTable.size)
    val neighborElems = new Array[IElement](neighborTable.size)
    neighborTable.zipWithIndex.foreach(e => {
      nodeIds(e._2) = e._1._1
      neighborElems(e._2) = new AliasTable(e._1._2._1, e._1._2._2)
    })

    graphs.getOrElse(nodeNames(0), null).psfUpdate(
      new InitAliasTableByName(new GeneralInitByNameParam(graphs.getOrElse(nodeNames(0), null).id,
      nodeIds, neighborElems, nodeNames(1)))).get()
    println(s"init ${neighborTable.length} alias table...")
  }

  def initNodeFeatures(keys: Array[Long], features: Array[Feature],
                       graphName: String, numBatch: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initNodeFeaturesByBatch(keys.slice(start, end), features.slice(start, end), graphName)
      start += step
    }
  }

  def initNodeFeaturesByBatch(batchKeys: Array[Long], batchFeatures: Array[Feature], graphName: String): Unit = {
    val graph = graphs.getOrElse(graphName, null.asInstanceOf[PSMatrix])  // choose userGraph or itemGraph
    val features = new Array[IElement](batchFeatures.length)
    for (i <- features.indices) {
      features(i) = batchFeatures(i)
    }

    val func = new InitMultiNodeFeats(new GeneralInitParam(graph.id, batchKeys, features))
    graph.asyncPsfUpdate(func).get()
  }

  def getFeatures(keys: Array[Long], graphName: String): Long2ObjectOpenHashMap[FloatVector] = {
    getFeatures(keys, graphName, FeatureFormat.DENSE)._1
  }

  def getFeatures(keys: Array[Long], graphName: String, featureFormat: FeatureFormat): (Long2ObjectOpenHashMap[FloatVector], Long2ObjectOpenHashMap[FloatVector]) = {
    val graph = graphs.getOrElse(graphName, null.asInstanceOf[PSMatrix])  // choose userGraph or itemGraph
    val res = graph.psfGet(new GetMultiNodeFeats(new GetNodeAttrsParam(graph.id, keys, featureFormat)))
      .asInstanceOf[GetNodeFeatsResult]
    (res.getnodeIdToFeats(0), res.getnodeIdToFeats(1))
  }

  def sampleFeatures(size: Int, graphName: String): Array[FloatVector] = {
    val graph = graphs.getOrElse(graphName, null.asInstanceOf[PSMatrix])  // choose userGraph or itemGraph
    val features = new Array[FloatVector](size)
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

  def sampleNeighbors(keys: Array[Long], count: Int, edgeName: String, sampleMethod: SampleMethod): Long2ObjectOpenHashMap[Neighbor] = {
    val nodes = edgeName.split("-")
    val graph = graphs.getOrElse(nodes(0), null.asInstanceOf[PSMatrix])
    val storageType = storageTypes.getOrElse(nodes(0), NeighborStorageType.LONGARRAY)

    graph.psfGet(new SampleNeighborByName(new SampleNeighborByNameParam(graph.id, keys, count, nodes(1), sampleMethod, storageType)))
      .asInstanceOf[SampleNeighborResult].getNodeIdToSampleNeighbors
  }

  def initEmbeddings(featureIds: Map[String, RDD[Long]], batchSize: Int, slots: Int, initMethod: String): Unit = {
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
  }

  def initEmbeddingsByBatch(embeddingName: String, nodeIds: Array[Long], seed: Int, numSlots: Int, featureNum: Long, embeddingDim: Int, initMethod: String): Unit = {
    val fin = featureNum
    val fout = embeddingDim.toLong
    val initFunc = InitUtils.apply(initMethod, fin, fout)
    val embedding = embeddings.getOrElse(embeddingName, null)
    val func = new UniversalEmbeddingInitAsNodes(
      new UniversalEmbeddingInitParam(embedding.id, nodeIds, seed, embeddingDim, numSlots, initFunc.getFloats(), initFunc.getInts()))
    embedding.psfUpdate(func).get()
  }

  def initExtraEmbeddings(keys: Array[Long], features: Array[IntFloatVector],
                          embeddingName: String, numBatch: Int, slot: Int): Unit = {
    val step = if (keys.length > numBatch) keys.length / numBatch else 1
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initExtraEmbeddingsByBatch(keys.slice(start, end), features.slice(start, end), embeddingName, slot)
      start += step
    }
  }

  def initExtraEmbeddingsByBatch(batchKeys: Array[Long], batchEmbeddings: Array[IntFloatVector], embeddingName: String, slot: Int): Unit = {
    val embedding = embeddings.getOrElse(embeddingName, null.asInstanceOf[PSMatrix])
    val dim = embedDims.getOrElse(embeddingName, -1)
    val features = new Array[IElement](batchEmbeddings.length)
    for (i <- features.indices) {
      features(i) = new EmbeddingOrGrad(batchEmbeddings(i).getStorage.getValues)
    }

    val func = new UniversalEmbeddingExtraInitAsNodes(new UniversalEmbeddingExtraInitParam(embedding.id, batchKeys, features, dim, slot))
    embedding.asyncPsfUpdate(func).get()
  }

  def getEmbedding(indices: Array[Long], graphType: String): Long2ObjectOpenHashMap[Array[Float]] = {
    val embedding = embeddings.getOrElse(graphType, null)
    val func = new UniversalEmbeddingGet(new GetNodeAttrsParam(embedding.id, indices))
    embedding.psfGet(func).asInstanceOf[GetFloatArrayAttrsResult].getNodeIdToContents
  }

  def updateEmbedding(grads: Array[Vector], optim: AsyncOptim, graphType: String): Unit = {
    val dim = embedDims.getOrElse(graphType, -1)
    optim.update(embeddings.getOrElse(graphType, null.asInstanceOf[PSMatrix]), dim, (0 until dim).toArray, grads)
  }


  def updateEmbedding(grads: Long2ObjectOpenHashMap[Array[Float]], optim: AsyncOptim, graphType: String): Unit = {
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

  override def loadFeatEmbed(featEmbedPath: String): Unit = {
    val ctx = new ModelLoadContext(featEmbedPath)
    embeddings.foreach{ case (name, _) =>
      val embeddingCtx = new MatrixLoadContext(name + "Embedding")
      ctx.addMatrix(embeddingCtx)
    }
    PSContext.getOrCreate(SparkContext.getOrCreate()).load(ctx)
  }

  /**
    * Dump the matrices on PS to HDFS
    *
    * @param checkpointId checkpoint id
    */
  override def checkpointMatrices(checkpointId: Int): Unit = {
    val matrixNames = new ArrayBuffer[String]()
    if (graphs != null) {
      graphs.keys.foreach(name => matrixNames.append(name))
    }
    matrixNames.append("weights")
    if (labels != null) {
      labels.keys.foreach(name => matrixNames.append(name))
    }
    if (testLabels != null) {
      testLabels.keys.foreach(name => matrixNames.append(name))
    }
    if (embeddings != null) {
      embeddings.keys.foreach(name => matrixNames.append(name))
    }
    CheckpointUtils.checkpoint(checkpointId, matrixNames.toArray)
  }

  def getNeighborStorageTypes(): Map[String,NeighborStorageType] = storageTypes
}

object HGNNPSModel {
  def apply(nodesMap: HashMap[String, (Long, Long)], featureIds: Option[Map[String, RDD[Long]]],
            weightSize: Int, optim: AsyncOptim, indexs: Map[String, RDD[Long]],
            embedDims: Map[String, Int], featureDims: Map[String, Long], psNumPartition: Int,
            useBalancePartition: Boolean, labelDF: Option[Map[String, DataFrame]],
            testLabelDF: Option[Map[String, DataFrame]], featureSplitIdxs: Map[String, Int]): HGNNPSModel = {
    val nodeNumMap = new HashMap[String, Long]()
    indexs.map{ case (name, rdd) =>
      var nodeNum = rdd.distinct().count()
      val num = nodeNumMap.getOrElse(name, -1L)
      nodeNum = if(nodeNum < num) num else nodeNum
      nodeNumMap.put(name, nodeNum)
    }

    val graphs = nodesMap.map{ case(name, (minId, maxId)) =>
      val num = nodeNumMap.getOrElse(name, -1L)
      val modelContext = new ModelContext(psNumPartition, minId, maxId, num, name,
        SparkContext.getOrCreate().hadoopConfiguration)
      val graphMatrixContext = ModelContextUtils.createMatrixContext(modelContext,
        RowType.T_ANY_LONGKEY_SPARSE, classOf[GraphNode])

      //todo: balance partition
      //      if (!modelContext.isUseHashPartition && useBalancePartition)
      //        LoadBalancePartitioner.partition(index, psNumPartition, modelContext)

      (name, PSMatrix.matrix(graphMatrixContext))
    }.toMap

    // create weights matrix context
    val weights = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weights.setRowType(RowType.T_FLOAT_DENSE)
    weights.setPartitionerClass(classOf[ColumnRangePartitioner])
    val weightsVec = PSVector.vector(weights)

    val labels = if (labelDF.nonEmpty) {
      labelDF.get.map{case (name, df) =>
        val (minId, maxId) = nodesMap.get(name).get
        val num = nodeNumMap.getOrElse(name, -1L)
        // create labels matrix context
        val labelsModelContext = new ModelContext(psNumPartition, minId, maxId, num, name + "Labels",
          SparkContext.getOrCreate().hadoopConfiguration)
        val labelsMatrixContext = ModelContextUtils.createMatrixContext(labelsModelContext, RowType.T_FLOAT_SPARSE_LONGKEY)
        (name, PSVector.vector(labelsMatrixContext))
      }
    } else null

    val testLabels = if (testLabelDF.nonEmpty) {
      testLabelDF.get.map{ case (name, df) =>
        val (minId, maxId) = nodesMap.get(name).get
        val num = nodeNumMap.getOrElse(name, -1L)
        // create testLabels matrix context
        val testLabelsModelContext = new ModelContext(psNumPartition, minId, maxId, num,
          name + "TestLabels", SparkContext.getOrCreate().hadoopConfiguration)
        val testLabelsMatrixContext = ModelContextUtils.createMatrixContext(testLabelsModelContext, RowType.T_FLOAT_SPARSE_LONGKEY)
        (name, PSVector.vector(testLabelsMatrixContext))
      }
    } else null

    val featureNumMap = new HashMap[String, Long]()
    if (featureIds.nonEmpty) {
      featureIds.get.map { case (name, rdd) =>
        var featureNum = rdd.count()
        val num = featureNumMap.getOrElse(name, -1L)
        featureNum = if (featureNum < num) num else featureNum
        featureNumMap.put(name, featureNum)
      }
    }

    val embeddings = if (embedDims.size > 1) {
      nodesMap.map{ case(name, (_, _)) =>
        val num = featureNumMap.getOrElse(name, -1L)
        val modelContext = new ModelContext(psNumPartition, 0, featureDims.getOrElse(name, -1L) + 1 - featureSplitIdxs.getOrElse(name, 0), num, name + "Embedding",
          SparkContext.getOrCreate().hadoopConfiguration)
        val matrixContext = ModelContextUtils.createMatrixContext(modelContext,
          RowType.T_ANY_LONGKEY_SPARSE, classOf[UniversalEmbeddingNode])

        (name, PSMatrix.matrix(matrixContext))
      }.toMap
    } else null

    val storageTypes = nodesMap.map{ case (name, (minId, maxId)) =>
      (name, if (minId > Int.MinValue && maxId < Int.MaxValue) NeighborStorageType.INTARRAY else NeighborStorageType.LONGARRAY)
    }.toMap

    new HGNNPSModel(graphs, weightsVec, labels, testLabels, embeddings, embedDims, storageTypes)
  }
}