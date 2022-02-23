package com.tencent.angel.pytorch.examples.supervised.cluster
import com.tencent.angel.pytorch.graph.gcn.GAMLP
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object GAMLPExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val featureInput = params.getOrElse("featurePath", "")
    val labelPath = params.getOrElse("labelPath", "")
    val testLabelPath = params.getOrElse("testLabelPath", "")
    val predictOutputPath = params.getOrElse("predictOutputPath", "")
    val embeddingPath = params.getOrElse("embeddingPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val fieldNum = params.getOrElse("fieldNum", "-1").toInt
    val featEmbedDim = params.getOrElse("featEmbedDim", "-1").toInt
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
    val batchSize = params.getOrElse("batchSize", "100").toInt
    var torchModelPath = params.getOrElse("upload_torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val featureDim = params.getOrElse("featureDim", "-1").toInt
    val optimizer = params.getOrElse("optimizer", "adam")
    var psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    var numPartitions = params.getOrElse("numPartitions", "10").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "2").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "3").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val testRatio = params.getOrElse("testRatio", "0.5").toFloat
    val format = params.getOrElse("format", "sparse")
    val numSamples = 1
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val numBatchInit = params.getOrElse("numBatchInit", "5").toInt
    val actionType = params.getOrElse("actionType", "train")
    val periods = params.getOrElse("periods", "1000").toInt
    val checkpointInterval = params.getOrElse("checkpointInterval", "0").toInt
    val decay = params.getOrElse("decay", "0.001").toFloat
    var evals = params.getOrElse("evals", "acc")
    val validatePeriods = params.getOrElse("validatePeriods", "5").toInt
    val useSecondOrder = false
    var useSharedSamples = params.getOrElse("useSharedSamples", "false").toBoolean
    if (batchSize < 128) useSharedSamples = false
    val saveCheckpoint = params.getOrElse("saveCheckpoint", "false").toBoolean
    val numLabels = params.getOrElse("numLabels", "1").toInt // a multi-label classification task if numLabels > 1
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    val sep = params.getOrElse("sep", "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }

    val labelSep = params.getOrElse("labelsep", "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }
    val hops = params.getOrElse("hops", "2").toInt

    if (numLabels > 1) evals = "multi_auc"

    val conf = new SparkConf()
    conf.set("spark.executor.extraLibraryPath", "./torch/torch-lib")
    conf.set("spark.executorEnv.OMP_NUM_THREADS", "2")
    conf.set("spark.executorEnv.MKL_NUM_THREADS", "2")
    val extraJavaOptions = conf.get("spark.executor.extraJavaOptions")
    conf.set("spark.executor.extraJavaOptions", extraJavaOptions +
      " -Djava.library.path=$JAVA_LIBRARY_PATH:/data/gaiaadmin/gaiaenv/tdwgaia/lib/native:.:./torch/torch-lib")

    val sc = new SparkContext(conf)

    //auto-adjust numPartitions and psNumPartition
    numPartitions = PartitionUtils.getDataPartitionNum(numPartitions, conf, numPartitionsFactor)
    psNumPartition = PartitionUtils.getPsPartitionNum(psNumPartition, conf, psNumPartitionFactor)
    println(s"numPartition: $numPartitions, numPsPartition: $psNumPartition")

    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)

    val gamlp = new GAMLP()
    gamlp.setTorchModelPath(torchModelPath)
    gamlp.setFeatureDim(featureDim * (hops + 1))
    gamlp.setOptimizer(optimizer)
    gamlp.setUseBalancePartition(false)
    gamlp.setBatchSize(batchSize)
    gamlp.setStepSize(stepSize)
    gamlp.setPSPartitionNum(psNumPartition)
    gamlp.setPartitionNum(numPartitions)
    gamlp.setUseBalancePartition(useBalancePartition)
    gamlp.setNumEpoch(numEpoch)
    gamlp.setStorageLevel(storageLevel)
    gamlp.setTestRatio(testRatio)
    gamlp.setDataFormat(format)
    gamlp.setNumSamples(numSamples)
    gamlp.setNumBatchInit(numBatchInit)
    gamlp.setPeriods(periods)
    gamlp.setCheckpointInterval(checkpointInterval)
    gamlp.setDecay(decay)
    gamlp.setEvaluations(evals)
    gamlp.setValidatePeriods(validatePeriods)
    gamlp.setUseSharedSamples(useSharedSamples)
    gamlp.setUseSecondOrder(useSecondOrder)
    gamlp.setSaveCheckpoint(saveCheckpoint)
    gamlp.setNumLabels(numLabels)
    gamlp.setBatchSizeMultiple(batchSizeMultiple)
    gamlp.setFeatEmbedPath(featureEmbedInputPath)
    gamlp.setFeatEmbedDim(featEmbedDim)
    gamlp.setFieldNum(fieldNum)
    gamlp.setFieldMultiHot(fieldMultiHot)

    val edges = GraphIO.load(edgeInput, isWeighted = false, sep = sep)
    val features = IOFunctions.loadFeature(featureInput, sep = "\t")
    val labels = if (labelPath.length > 0) {
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath, seq=labelSep))
    } else None
    val testLabels = if (testLabelPath.length > 0)
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(testLabelPath, sep = "p") else IOFunctions.loadLabel(testLabelPath, seq=labelSep))
    else None

    val (model, graph) = gamlp.initialize(edges, features, labels, testLabels)
    gamlp.showSummary(model, graph)

    if (actionType == "train")
      gamlp.fit(model, graph, outputModelPath)

    if (predictOutputPath.length > 0) {
      val predict = gamlp.genLabels(model, graph)
      GraphIO.save(predict, predictOutputPath, seq = " ")
    }

    if (embeddingPath.length > 0) {
      val embedding = gamlp.genEmbedding(model, graph)
      GraphIO.save(embedding, embeddingPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      gamlp.save(model, outputModelPath)
      if (fieldNum > 0) {
        gamlp.saveFeatEmbed(model, outputModelPath)
      }
    }

    PSContext.stop()
    sc.stop()
  }
}
