package com.tencent.angel.pytorch.graph.gcn


import com.tencent.angel.graph.data.FeatureFormat
import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.eval.Evaluation
import com.tencent.angel.pytorch.io.{DataLoaderUtils, IOFunctions}
import com.tencent.angel.pytorch.params._
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.graph.utils.params._
import com.tencent.angel.pytorch.params._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import com.tencent.angel.spark.ml.util.LogUtils
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import Array._
import scala.collection.mutable


class GATNE extends HGNN with HasWindowSize with HasNegative with HasNodeTypes with HasEdgeTypes with HasInitMethod
  with HasSchema with HasContextDim with HasValidate with HasMaxIndex with HasEachNumSample with HasLogStep
  with HasModelSaveInterval with HasEmbeddingSaveInterval with HasFeatureSplitIdxs with HasContextEmbedPath
  with HasNegSampleByNodeType with HasLocalNegativeSample {
  val minIndexMap: mutable.Map[Int, Int] = mutable.Map[Int, Int]()
  val maxIndexMap: mutable.Map[Int, Int] = mutable.Map[Int, Int]()


  def initialize(edgeDF: DataFrame,
                 featureDFs: Map[Int, Dataset[Row]],
                 nodeTypeRDD: RDD[(Long, Int)],
                 mean: Float,
                 std: Float): EmbeddingGNNPSModel = {
    val start = System.currentTimeMillis()

    val (minId, maxId, numNodes) = getMinMaxNodeId(nodeTypeRDD.map(r => r._1))
    println(s"minId=$minId maxId=$maxId numNodes=$numNodes")
    setMaxIndex(nodeTypeRDD.count().toInt)

    // create weights, graph on servers
    TorchModel.setPath($(torchModelPath))
    val torch = TorchModel.get()
    val weightsSize = torch.getParametersTotalSize
    println(s"weight total size=$weightsSize")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    setNodeTypes(torch.getNodeTypes)
    setEdgeTypes(torch.getEdgeTypes)
    setSchema(torch.getSchema)
    setFeatureDims(torch.getFeatureDims)
    setFieldNums(torch.getFieldNums)
    setEmbedDims(torch.getFeatEmbedDims)
    if ($(dataFormat) == FeatureFormat.DENSE_HIGH_SPARSE) setFeatureSplitIdxs(torch.getFeatureSplitIdxs)

    // init ps model
    val featureIds = if ($(fieldNums).nonEmpty) Option(getFeatureIndexs(featureDFs.map(f => (f._1.toString, f._2))).
      map(f => (f._1.toInt, f._2))) else None
    if (featureIds.nonEmpty) featureIds.get.foreach(_._2.persist($(storageLevel)))

    val model = EmbeddingGNNPSModel.apply(minId, maxId + 1, weightsSize, getOptimizer,
      numNodes, featureIds, getEmbedDimsByInt, getFeatureDimsByInt, getFeatureSplitIdxsByInt, $(psPartitionNum), $(useBalancePartition))

    model.setWeights(torch.getParameters)
    TorchModel.put(torch)

    if ($(featEmbedPath).nonEmpty) {
      initExtraEmbeddings(model, $(featEmbedPath), featureDFs.keys.toArray, featureIds.get)
    } else {
      if (torch.getFieldNums.nonEmpty)
        model.asInstanceOf[EmbeddingGNNPSModel].initEmbeddings(featureIds.get, $(batchSize), getOptimizer.getNumSlots(), getInitMethod)
    }

    makeGraph(edgeDF, nodeTypeRDD, model)
    initFeatures(model, featureDFs)
    initNodeTypes(model, nodeTypeRDD)
    initIndex2Node(model, nodeTypeRDD, $(negSampleByNodeType))

    if ($(contextEmbedPath).nonEmpty) {
      initExtraContext(model, nodeTypeRDD, numNodes, mean, std, $(contextEmbedPath))
    } else {
      initContext(model, nodeTypeRDD, numNodes, Random.nextInt(), mean, std)
    }

    val end = System.currentTimeMillis()
    println(s"initialize cost ${(end - start) / 1000}s")
    val startTs = System.currentTimeMillis()
    if ($(saveCheckpoint))
      model.checkpointMatrices(0)
    println(s"Write checkpoint use time=${(System.currentTimeMillis() - startTs)}ms")
    model
  }

  def getMinMaxNodeId(nodeRDD: RDD[Long]): (Long, Long, Long) =
    nodeRDD.mapPartitions(DataLoaderUtils.summarizeApplyOpByNode)
      .reduce(DataLoaderUtils.summarizeReduceOp)

  def makeGraph(edges: DataFrame, nodeTypeRDD: RDD[(Long, Int)], model: EmbeddingGNNPSModel): Unit = {
    val adjGraph = makePartition(edges, nodeTypeRDD, 0, 1)
    adjGraph.persist($(storageLevel))
    adjGraph.mapPartitions(iterator => Iterator.single(model.initNeighbors(iterator.toArray, $(numBatchInit)))).count()
    adjGraph.unpersist(true)
  }

  def makePartition(edges: DataFrame, nodeTypeRDD: RDD[(Long, Int)], start: Int = 0, end: Int = 1): RDD[(Long, Iterable[(Long, Int)])] = {
    var model_schema: Map[(Int, Int), Int] = Map()
    getSchema.split(",").map(s => s.split("-")).foreach(f => {
      model_schema += ((f(0).toInt, f(1).toInt) -> f(2).toInt)
      model_schema += ((f(2).toInt, f(1).toInt) -> f(0).toInt)
    })

    edges.select("src", "dst", "type").rdd
      .map(row => (row.getLong(start), (row.getLong(end), row.getInt(2))))
      .union(edges.select("dst", "src", "type").rdd
        .map(row => (row.getLong(start), (row.getLong(end), row.getInt(2)))))
      .join(nodeTypeRDD)                                                      // src, ((dst, edge_type), src_type)
      .map(r => (r._2._1._1, (r._1, r._2._2, r._2._1._2)))                    // dst, (src, src_type, edge_type)
      .join(nodeTypeRDD)                                                      // dst, ((src, src_type, edge_type), dst_type)
      .map(r => (r._2._1._1, r._1, r._2._1._2, r._2._1._3, r._2._2))          // src, dst, src_type, edge_type, dst_type
      .filter(r => model_schema.getOrElse((r._3, r._4), -1) == r._5)
      .map(r => (r._1, (r._2, r._4)))
      .groupByKey($(partitionNum))
  }

  def initFeatures(model: EmbeddingGNNPSModel,
                   features: Map[Int, Dataset[Row]]): Unit = {
    val dimsMap = getFeatureDimsByInt
    val featureSplitIdxsMap = getFeatureSplitIdxsByInt
    features.foreach{ case (name, feat) =>
      feat.select("node", "feature").rdd.filter(row => row.length > 0)
        .filter(row => row.get(0) != null)
        .map(row => (row.getLong(0), row.getString(1)))
        .map(f => (f._1, SampleParser.parseFeature(f._2, dimsMap.getOrElse(name, 0).toLong, $(dataFormat), featureSplitIdxsMap.getOrElse(name, 0))))
        .repartition($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator.single(NodeFeaturePartition.apply(index, it)))
        .map(_.init(model, $(numBatchInit))).count()
    }
  }

  def initExtraEmbeddings(model: EmbeddingGNNPSModel,
                          featEmbedPath: String,
                          nodeNames: Array[Int],
                          featureIds: Map[Int, RDD[Long]]): Unit = {
    model.initEmbeddings(featureIds, $(batchSize), getOptimizer.getNumSlots(), getInitMethod)

    val conf = SparkHadoopUtil.get.newConfiguration(SparkContext.getOrCreate().getConf)
    val keyValueSep = IOFunctions.parseSep(conf.get("angel.embedding.keyvalue.sep", "colon"))
    val featSep = IOFunctions.parseSep(conf.get("angel.embedding.feature.sep", "space"))

    val beforeRandomize = System.currentTimeMillis()
    nodeNames.foreach{ name =>
      val path = featEmbedPath + "/" + name + "Embedding"
      IOFunctions.loadFeature(path, sep = keyValueSep).rdd
        .map(r => (r.getLong(0), SampleParser.parseEmbedding(r.getString(1), featSep)))
        .repartition($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator.single(NodeFeaturePartition.apply(index, it)))
        .map(_.init(model, name, $(numBatchInit), getOptimizer.getNumSlots())).count()
    }
    LogUtils.logTime(s"feature embedding successfully initialized by extra embedding, cost ${(System.currentTimeMillis() - beforeRandomize) / 1000.0}s")
  }

  def initContext(model: EmbeddingGNNPSModel,
                  nodeTypeRDD: RDD[(Long, Int)],
                  numNodes: Long,
                  seed: Int,
                  mean: Float,
                  std: Float): Unit={
    val beforeRandomize = System.currentTimeMillis()
    nodeTypeRDD.map(r => r._1).mapPartitions { iterator =>
      iterator.sliding(getBatchSize, getBatchSize)
        .foreach(batch => model.initContext(batch.toArray, $(contextDim), $(initMethod), numNodes, getOptimizer, seed, mean, std))
      Iterator.single()
    }.count()
    LogUtils.logTime(s"Context successfully Randomized, cost ${(System.currentTimeMillis() - beforeRandomize) / 1000.0}s")
  }

  def initExtraContext(model: EmbeddingGNNPSModel,
                       nodeTypeRDD: RDD[(Long, Int)],
                       numNodes: Long,
                       mean: Float,
                       std: Float,
                       extraContextInputPath: String): Unit = {
    initContext(model, nodeTypeRDD, numNodes, Random.nextInt(), mean, std)

    val conf = SparkHadoopUtil.get.newConfiguration(SparkContext.getOrCreate().getConf)
    val keyValueSep = IOFunctions.parseSep(conf.get("angel.embedding.keyvalue.sep", "colon"))
    val featSep = IOFunctions.parseSep(conf.get("angel.embedding.feature.sep", "space"))

    val beforeRandomize = System.currentTimeMillis()
    val path = extraContextInputPath + "/" + "context"
    IOFunctions.loadFeature(path, sep = keyValueSep).rdd
      .map(r => (r.getLong(0), SampleParser.parseEmbedding(r.getString(1), featSep)))
      .repartition($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(NodeFeaturePartition.apply(index, it)))
        .map(_.initContext(model, $(contextDim), $(numBatchInit), getOptimizer.getNumSlots())).count()
    LogUtils.logTime(s"Context successfully initialized by extra embedding, cost ${(System.currentTimeMillis() - beforeRandomize) / 1000.0}s")
  }

  def initNodeTypes(model: EmbeddingGNNPSModel, nodeTypeRDD: RDD[(Long, Int)]): Unit = {
    nodeTypeRDD.mapPartitionsWithIndex((index, it) =>
      Iterator.single(NodeTypePartition.apply(index, it, model.dim)))
      .map(_.init(model)).count()
  }

  def initIndex2Node(model: EmbeddingGNNPSModel, nodeTypeRDD: RDD[(Long, Int)], negSampleByNodeType: Boolean): Unit = {
    if (negSampleByNodeType) {
      val all_nodeTypes = getNodeTypes.split(",").map(_.toInt)
      var cur_node_num = 0
      all_nodeTypes.foreach(t => {
        val temp = nodeTypeRDD.filter(r => r._2 == t).map(r => r._1)
        val num = temp.count().toInt
        minIndexMap += t -> cur_node_num
        maxIndexMap += t -> (cur_node_num + num)
        println(s"node type = ${t}, minIndex: ${minIndexMap.getOrElse(t, 0)} maxIndex: ${maxIndexMap.getOrElse(t, 0)}")
        temp.zipWithIndex().map(r => (r._2.toInt + cur_node_num, r._1))
          .mapPartitionsWithIndex((index, it) =>
            Iterator.single(Index2NodePartition.apply(index, it, model.dim)))
          .map(_.init(model)).count()
        cur_node_num += num
      })
    }
    else {
      nodeTypeRDD.map(r => r._1).zipWithIndex().map(r => (r._2.toInt, r._1))
        .mapPartitionsWithIndex((index, it) =>
          Iterator.single(Index2NodePartition.apply(index, it, model.dim)))
        .map(_.init(model)).count()
    }
  }

  def showSummary(model: GNNPSModel): Unit = {
    println(s"nnzFeatures=${model.nnzFeatures()}")
    println(s"nnzNode=${model.nnzNodes()}")
    println(s"nnzEdge=${model.nnzEdge()}")
    println(s"nnzNeighbor=${model.nnzNeighbors()}")
  }

  def generateTrainPairs(model: EmbeddingGNNPSModel,
                         trainRDD: RDD[Array[Long]]): RDD[GATNEPartition] = {
    println(s"total walk sequence ${trainRDD.count()}")
    val start = System.currentTimeMillis()
    val trainPairs = trainRDD.mapPartitions { iterator =>
      iterator.sliding(getBatchSize, getBatchSize)
        .map(batch => parseBatchData(batch.toArray, getWindowSize, model))
    }.flatMap(r => r).persist($(storageLevel))

    val len = trainPairs.count()
    val end = System.currentTimeMillis()
    println(s"generate train pairs cost ${(end - start) / 1000}s, total train pars ${len}")

    trainPairs.mapPartitionsWithIndex((index, it) =>
        Iterator.single(GATNEPartition.apply(index, it, $(torchModelPath), $(dataFormat))))
  }

  def generateTestPairs(testEdges: DataFrame, nodeTypeRDD: RDD[(Long, Int)]): RDD[GATNEPartition] = {
    val nodes = testEdges.select("src", "type").rdd.map(f => (f.getLong(0), f.getInt(1)))
      .union(testEdges.select("dst", "type").rdd.map(f => (f.getLong(0), f.getInt(1))))
      .distinct()
    val nodeWithType = nodeTypeRDD.join(nodes).map(f => (f._1, f._1, f._2._1, f._2._2))       // node, node, node_type, edge_type
    nodeWithType.mapPartitionsWithIndex((index, it) =>
      Iterator.single(GATNEPartition.apply(index, it, $(torchModelPath), $(dataFormat))))
  }

  def generatePredictPairs(nodeTypeRDD: RDD[(Long, Int)]): RDD[GATNEPartition] = {
    val all_edgeTypes = getEdgeTypes.split(",").map(_.toInt)
    nodeTypeRDD.flatMap(f => all_edgeTypes.map(e => (f._1, f._1, f._2, e))).mapPartitionsWithIndex((index, it) =>
      Iterator.single(GATNEPartition.apply(index, it, $(torchModelPath), $(dataFormat))))
  }

  def parseBatchData(sentences: Array[Array[Long]],
                     windowSize: Int,
                     model: EmbeddingGNNPSModel): Array[(Long, Long, Int, Int)] = {
    var pairs = new ArrayBuffer[(Long, Long, Int, Int)]()

    val nodes = sentences.flatten.distinct
    val node2Type = model.readNodeTypes(nodes)

    for (s <- sentences.indices) {
      val sentence = sentences(s)
      val cur_edge_type = sentence(0).toInt
      for (srcIndex <- 1 until sentence.length) {
        var dstIndex = Math.max(srcIndex - windowSize, 1)
        while (dstIndex < Math.min(srcIndex + windowSize + 1, sentence.length)) {
          if (srcIndex != dstIndex) {
            val src = sentence(srcIndex)
            val dst = sentence(dstIndex)
            pairs.append((src, dst, node2Type.get(src), cur_edge_type))
          }
          dstIndex += 1
        }
      }
    }
    pairs.toArray
  }

  def fit(model: EmbeddingGNNPSModel,
          corpus: RDD[Array[Long]],
          testEdges: Option[DataFrame],
          nodeTypeRDD: RDD[(Long, Int)],
          outputModelPath: String,
          evaluateByEdgeType: Boolean): Unit = {
    val trainPairs = generateTrainPairs(model, corpus).persist($(storageLevel))
    trainPairs.foreachPartition(_ => Unit)

    val testPairs = if (testEdges.nonEmpty) Option(generateTestPairs(testEdges.get, nodeTypeRDD).persist($(storageLevel))) else None

    val optim = getOptimizer
    println(s"optimizer: $optim")
    println(s"evals: ${getEvaluations.mkString(",")}")
    val all_nodeTypes = getNodeTypes.split(",").map(_.toInt)
    val all_edgeTypes = getEdgeTypes.split(",").map(_.toInt)
    var schema: Map[(Int, Int), Int] = Map()
    getSchema.split(",").map(s => s.split("-")).map(f => {
      schema += ((f(0).toInt, f(1).toInt) -> f(2).toInt)
      schema += ((f(2).toInt, f(1).toInt) -> f(0).toInt)
      f
    })

    var startTs = System.currentTimeMillis()
    for (curEpoch <- 1 to getNumEpoch) {
      startTs = System.currentTimeMillis()
      val (lossSum, numSteps, allSteps) = trainPairs.map (_.asInstanceOf[GATNEPartition]
          .trainEpoch(model, $(batchSize), optim, all_nodeTypes, all_edgeTypes, schema, getFeatureDimsByInt, getEmbedDimsByInt,
            getFieldNumsByInt, getFeatureSplitIdxsByInt, $(fieldMultiHot), $(contextDim), $(negative), getEachNumSample,
            $(sampleMethod), $(logStep), $(localNegativeSample), $(negSampleByNodeType), $(maxIndex), maxIndexMap, minIndexMap))
      .reduce((f1, f2) => (f1._1 + f2._1, math.max(f1._2, f2._2), f1._3 + f2._3))

      var ends = System.currentTimeMillis()
      print(s"curEpoch=$curEpoch lr=${optim.getCurrentEta.toFloat} trainLoss=${lossSum / allSteps} train cost=${(ends - startTs) / 1000}s ")
      optim.step(numSteps)
      println()

      if (testPairs.nonEmpty && (curEpoch % $(validatePeriods) == 0 || curEpoch == $(numEpoch))) {
        val start = System.currentTimeMillis()
        val testNodeEmbedding = genEmbedding(model, testPairs.get)
        evaluate(testNodeEmbedding, testEdges.get, evaluateByEdgeType)
        print(s" test cost=${(System.currentTimeMillis() - start) / 1000}s ")
        println()
      }

      if (outputModelPath.nonEmpty && (curEpoch % $(saveModelInterval) == 0 || curEpoch == $(numEpoch)))
        save(model, outputModelPath, curEpoch)
    }
  }

  def evaluate(nodeEmb: RDD[(Long, Int, Array[Float])],
               testEdges: DataFrame,
               evaluateByEdgeType: Boolean): Unit = {
    val scores = testEdges.select("src", "dst", "type", "label").rdd
        .map(f => ((f.getLong(0), f.getInt(2)), (f.getLong(1), f.getFloat(3))))   // (src,type),(dst,label)
        .join(nodeEmb.map(f => ((f._1, f._2), f._3)), getPartitionNum)            // (src,type),((dst,label),src_emb)
        .map(f => ((f._2._1._1, f._1._2), (f._2._1._2, f._2._2)))                 // (dst,type),(label,src_emb)
        .join(nodeEmb.map(f => ((f._1, f._2), f._3)), getPartitionNum)            // (dst,type),((label,src_emb),dst_emb)
        .map(f => (f._1._2, f._2._1._1, regularizeDot(f._2._1._2, f._2._2)))               // (type,label,score)
        .persist(getStorageLevel)

    if (evaluateByEdgeType) {
      val all_edgeTypes = getEdgeTypes.split(",").map(_.toInt)
      all_edgeTypes.foreach {e => {
        val e_scores = scores.filter(f => f._1 == e)
        if (!e_scores.isEmpty()) {
          val validateMetrics = Evaluation.eval(getEvaluations, e_scores.map(r => (r._2, r._3))).map(x => (x._1, x._2.toString))
          validateMetrics.foreach(f => println(s"\t\t\t\t\t\t\t\tedge_type ${e}: validate_${f._1}=${f._2} "))
        }
      }}
    }
    val validateMetrics = Evaluation.eval(getEvaluations, scores.map(r => (r._2, r._3))).map(x => (x._1, x._2.toString))
    validateMetrics.foreach(f => print(s"\t\t\t\t\t\t\t\tedge_type all: validate_${f._1}=${f._2} "))

    nodeEmb.unpersist()
    scores.unpersist()
  }

  def regularizeDot(x: Array[Float], y: Array[Float]): Float = {
    var dotValue = 0.0f
    var x_norm = 0.0f
    var y_norm = 0.0f
    x.indices.foreach(i => (dotValue += x(i) * y(i), x_norm += x(i) * x(i), y_norm += y(i) * y(i)))
    dotValue / (math.sqrt(x_norm).toFloat * math.sqrt(y_norm).toFloat)
  }

  def genEmbedding(model: EmbeddingGNNPSModel,
                   testPairs: RDD[GATNEPartition]): RDD[(Long, Int, Array[Float])] = {

    val all_nodeTypes = getNodeTypes.split(",").map(_.toInt)
    val all_edgeTypes = getEdgeTypes.split(",").map(_.toInt)
    var model_schema: Map[(Int, Int), Int] = Map()
    getSchema.split(",").map(s => s.split("-")).map(f => {
      model_schema += ((f(0).toInt, f(1).toInt) -> f(2).toInt)
      model_schema += ((f(2).toInt, f(1).toInt) -> f(0).toInt)
      f
    })

    testPairs.flatMap(_.asInstanceOf[GATNEPartition]
        .genEmbeddingEpoch(model, getBatchSize, all_nodeTypes, all_edgeTypes, model_schema, getFeatureDimsByInt, getEmbedDimsByInt,
          getFieldNumsByInt, getFeatureSplitIdxsByInt, $(fieldMultiHot), $(contextDim), $(negative), getEachNumSample, $(sampleMethod)))
  }

  def saveFeatEmbeds_(model: EmbeddingGNNPSModel, savePath: String, curEpoch: Int = -1): Unit = {
    if (getFieldNums.nonEmpty) {
      val path = if (curEpoch < 0) savePath + "/featEmbed" else savePath + s"/featEmbed_$curEpoch"
      model.saveFeatEmbed(path)
    }
  }

  def saveContext(model: EmbeddingGNNPSModel, savePath: String, curEpoch: Int = -1): Unit = {
    val path = if (curEpoch < 0) savePath + "/context" else savePath + s"/context_$curEpoch"
    model.saveContext(path)
  }
}
