package com.tencent.angel.pytorch.examples.unsupervised.cluster

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.graph.utils.GraphIO
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

object GATNEExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val testEdgeInput = params.getOrElse("testEdgePath", "")
    val featureInputs = params.getOrElse("featurePaths", "") //Feature files for each node type, 0:path,1:path,2:path, ...
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val ContextEmbeddingInputPath = params.getOrElse("ContextEmbeddingInputPath", "")
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
    val walkInput = params.getOrElse("walkPath", "")
    val nodeTypeInput = params.getOrElse("nodeTypePath", null)
    val embeddingOutputPath = params.getOrElse("embeddingOutputPath", "")
    var torchModelPath = params.getOrElse("torchModelPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val optimizer = params.getOrElse("optimizer", "adam")
    val contextDim = params.getOrElse("contextDim", "32").toInt
    val windowSize = params.getOrElse("window", "10").toInt
    val eachNumSamples = params.getOrElse("eachNumSamples", "")
    val numNegSamples = params.getOrElse("negative", "10").toInt
    val iniFuncType = params.getOrElse("iniFuncType", "randomNormal")
    val logStep = params.getOrElse("logStep", "1000").toInt
    val mean = params.getOrElse("mean", "0.0").toFloat
    val std = params.getOrElse("std", "1.0").toFloat
    val numEpoch = params.getOrElse("numEpoch", "5").toInt
    val stepSize = params.getOrElse("stepSize", "0.1").toFloat
    val decay = params.getOrElse("decay", "0.0").toFloat
    val actionType = params.getOrElse("actionType", "train")
    val negSampleByNodeType = params.getOrElse("negSampleByNodeType", "false").toBoolean
    val evaluateByEdgeType = params.getOrElse("evaluateByEdgeType", "false").toBoolean
    val outputEmbeddingByNodeType = params.getOrElse("outputEmbeddingByNodeType", "false").toBoolean
    val format = params.getOrElse("format", "dense")
    val batchSize = params.getOrElse("batchSize", "50").toInt
    var psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    var numPartitions = params.getOrElse("numPartitions", "10").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "2").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "3").toInt
    val validatePeriods = params.getOrElse("validatePeriods", "1").toInt
    val evals = params.getOrElse("evals", "auc")
    val saveCheckpoint = params.getOrElse("saveCheckpoint", "false").toBoolean
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_AND_DISK")
    val saveModelInterval = params.getOrElse("saveModelInterval", "2").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val isWeighted = params.getOrElse("isWeighted", "false").toBoolean
    val sampleMethod = if (isWeighted) params.getOrElse("sampleMethod", "random") else "random"
    val localNegativeSample = params.getOrElse("localNegativeSample", "false").toBoolean
    val sep = params.getOrElse("sep", "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }
    val featureSep = params.getOrElse("featureSep", "tab") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }

    val conf = new SparkConf()
    conf.set("spark.executor.extraLibraryPath", "./torch/torch-lib")
    conf.set("spark.executorEnv.OMP_NUM_THREADS", "2")
    conf.set("spark.executorEnv.MKL_NUM_THREADS", "2")
    val extraJavaOptions = conf.get("spark.executor.extraJavaOptions")
    conf.set("spark.executor.extraJavaOptions", extraJavaOptions +
      " -Djava.library.path=$JAVA_LIBRARY_PATH:/data/gaiaadmin/gaiaenv/tdwgaia/lib/native:.:./torch/torch-lib")
    conf.set("spark.hadoop." + AngelConf.ANGEL_PS_BACKUP_AUTO_ENABLE, "false")

    val sc = new SparkContext(conf)

    //auto-adjust numPartitions and psNumPartition
    numPartitions = PartitionUtils.getDataPartitionNum(numPartitions, conf, numPartitionsFactor)
    psNumPartition = PartitionUtils.getPsPartitionNum(psNumPartition, conf, psNumPartitionFactor)
    println(s"numPartition: $numPartitions, numPsPartition: $psNumPartition")

    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)

    val gatne = new GATNE()
    gatne.setTorchModelPath(torchModelPath)
    gatne.setOptimizer(optimizer)
    gatne.setDataFormat(format)
    gatne.setBatchSize(batchSize)
    gatne.setStepSize(stepSize)
    gatne.setValidatePeriods(validatePeriods)
    gatne.setPSPartitionNum(psNumPartition)
    gatne.setPartitionNum(numPartitions)
    gatne.setUseBalancePartition(useBalancePartition)
    gatne.setNumEpoch(numEpoch)
    gatne.setStorageLevel(storageLevel)
    gatne.setSaveCheckpoint(saveCheckpoint)
    gatne.setDecay(decay)
    gatne.setEvaluations(evals)
    gatne.setHasWeighted(isWeighted)
    gatne.setSampleMethod(sampleMethod)
    gatne.setContextDim(contextDim)
    gatne.setWindowSize(windowSize)
    gatne.setNegative(numNegSamples)
    gatne.setNegSampleByNodeType(negSampleByNodeType)
    gatne.setLocalNegativeSample(localNegativeSample)
    gatne.setSaveCheckpoint(saveCheckpoint)
    gatne.setSaveModelInterval(saveModelInterval)
    gatne.setInitMethod(iniFuncType)
    gatne.setEachNumSample(eachNumSamples)
    gatne.setLogStep(logStep)
    gatne.setFieldMultiHot(fieldMultiHot)
    gatne.setFeatEmbedPath(featureEmbedInputPath)
    gatne.setContextEmbedPath(ContextEmbeddingInputPath)

    val data = GraphIO.loadString(walkInput)
    val edges = IOFunctions.loadEdge(edgeInput, isTyped = true, sep = sep)
    val testEdges = if (testEdgeInput.nonEmpty) Option(IOFunctions.loadEdgeWithLabel(testEdgeInput, isTyped = true, sep = sep)) else None
    val nodes = edges.select("src", "dst").rdd.flatMap(r => Iterator((r.getLong(0), 1), (r.getLong(1), 1))).distinct()

    val featuresMap = featureInputs.split(",").map { pair =>
      val kv = pair.split(":", 2) // name:path
      (kv(0).toInt, IOFunctions.loadFeature(kv(1), sep = featureSep))
    }.toMap
    val nodeTypeRDD = IOFunctions.loadNodeType(nodeTypeInput, sep = sep)
      .select("node", "type")
      .rdd
      .map(r => (r.getLong(0), r.getInt(1)))
      .distinct()
      .join(nodes)
      .map(r => (r._1, r._2._1))
      .repartition(numPartitions)
      .persist(StorageLevel.fromString(storageLevel))

    val model = gatne.initialize(edges, featuresMap, nodeTypeRDD, mean, std)
    edges.unpersist()
    gatne.showSummary(model)

    val corpus = data.filter(f => f != null && f.nonEmpty)
      .map(f => f.stripLineEnd.split("[\\s+|,]").map(s => s.toLong))
      .filter(arr => arr.length > 1)
      .repartition(numPartitions)
      .persist(StorageLevel.fromString(storageLevel))

    if (actionType == "train")
      gatne.fit(model, corpus, testEdges, nodeTypeRDD, outputModelPath, evaluateByEdgeType)

    if (embeddingOutputPath.nonEmpty) {
      val spark = SparkSession.builder.getOrCreate
      var start = System.currentTimeMillis()
      val predictPairs = gatne.generatePredictPairs(nodeTypeRDD)

      val schema = StructType(Seq(
        StructField("node", LongType, nullable = false),
        StructField("edgeType", IntegerType, nullable = false),
        StructField("embedding", StringType, nullable = false)
      ))
      val all_embedding = gatne.genEmbedding(model, predictPairs).persist(StorageLevel.fromString(storageLevel))
      val c = all_embedding.count()
      if (outputEmbeddingByNodeType) {
        val all_nodeTypes = gatne.getNodeTypes.split(",").map(_.toInt)
        val embedding = all_embedding.map(r => (r._1, (r._2, r._3)))
        val nodeWithemb = nodeTypeRDD.join(embedding)
          .map(r => (r._1, r._2._1, r._2._2._1, r._2._2._2)) // node_id, node_type, edge_type, emb

        all_nodeTypes.foreach(t => {
          start = System.currentTimeMillis()
          val sub_embedding = nodeWithemb.filter(f => f._2 == t)
            .map(f => Row.fromSeq(Seq[Any](f._1, f._3, f._4.mkString(" "))))
          val sub_embeddingDF = spark.createDataFrame(sub_embedding, schema)
          GraphIO.save(sub_embeddingDF, embeddingOutputPath + "/" + t.toString, seq = "\t")
          println(s"gen embedding for type ${t}, cost ${System.currentTimeMillis() - start}ms")
        })
      } else {
        val embedding = all_embedding.map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3.mkString(" "))))
        val embeddingDF = spark.createDataFrame(embedding, schema)
        GraphIO.save(embeddingDF, embeddingOutputPath, seq = "\t")
        println(s"gen embedding for ${c} nodes of all types, cost ${System.currentTimeMillis() - start}ms")
      }
    }

    if (actionType == "train" && outputModelPath.nonEmpty) {
      gatne.save(model, outputModelPath)
      gatne.saveFeatEmbeds_(model, outputModelPath)
      gatne.saveContext(model, outputModelPath)
    }

    PSContext.stop()
    sc.stop()
  }
}
