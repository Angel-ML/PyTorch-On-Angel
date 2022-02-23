package com.tencent.angel.pytorch.examples.supervised.cluster

import com.tencent.angel.pytorch.graph.gcn.GCN
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel.MEMORY_ONLY

import scala.language.existentials

object AggregatorExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val featureInput = params.getOrElse("featurePath", "")
    val predictOutputPath = params.getOrElse("predictOutputPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val fieldNum = params.getOrElse("fieldNum", "-1").toInt
    val featEmbedDim = params.getOrElse("featEmbedDim", "-1").toInt
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
    val batchSize = params.getOrElse("batchSize", "100").toInt
    var torchModelPath = params.getOrElse("upload_torchModelPath", "model.pt")
    val featureDim = params.getOrElse("featureDim", "-1").toInt
    var psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    var numPartitions = params.getOrElse("numPartitions", "10").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "2").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "3").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val format = params.getOrElse("format", "sparse")
    val numSamples = params.getOrElse("samples", "5").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val numBatchInit = params.getOrElse("numBatchInit", "5").toInt
    val checkpointInterval = params.getOrElse("checkpointInterval", "0").toInt
    val useSecondOrder = false
    var useSharedSamples = params.getOrElse("useSharedSamples", "false").toBoolean
    if (batchSize < 128) useSharedSamples = false
    val saveCheckpoint = params.getOrElse("saveCheckpoint", "false").toBoolean
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    val sep = params.getOrElse("sep", "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }
    val hops = params.getOrElse("hops", "2").toInt

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

    /* Indeed, gcn is the aggregator here,
    but we obtain the aggregated/smoothed features with the help of
    method "genEmbedding" of class "GCN" */
    val gcn = new GCN()
    gcn.setTorchModelPath(torchModelPath)
    gcn.setFeatureDim(featureDim)
    gcn.setUseBalancePartition(false)
    gcn.setBatchSize(batchSize)
    gcn.setPSPartitionNum(psNumPartition)
    gcn.setPartitionNum(numPartitions)
    gcn.setUseBalancePartition(useBalancePartition)
    gcn.setStorageLevel(storageLevel)
    gcn.setDataFormat(format)
    gcn.setNumSamples(numSamples)
    gcn.setNumBatchInit(numBatchInit)
    gcn.setCheckpointInterval(checkpointInterval)
    gcn.setUseSharedSamples(useSharedSamples)
    gcn.setUseSecondOrder(useSecondOrder)
    gcn.setSaveCheckpoint(saveCheckpoint)
    gcn.setBatchSizeMultiple(batchSizeMultiple)
    gcn.setFeatEmbedPath(featureEmbedInputPath)
    gcn.setFeatEmbedDim(featEmbedDim)
    gcn.setFieldNum(fieldNum)
    gcn.setFieldMultiHot(fieldMultiHot)

    val edges = GraphIO.load(edgeInput, isWeighted = false, sep = sep)
    var features = IOFunctions.loadFeature(featureInput, sep = "\t")

    val (model, graph) = gcn.initialize(edges, features, None)

    assert(predictOutputPath.nonEmpty)
    val (minId, maxId, _) = gcn.getMinMaxId(edges)

    var featureList = features
    val spark = SparkSession.builder.getOrCreate

    for (hop <- 1 to hops) {
      gcn.initFeatures(model, features, minId, maxId)
      features = gcn.genEmbedding(model, graph)
      features = features.toDF("node", "feature").persist(MEMORY_ONLY)
      
      val rdd_feature_merged = featureList
        .join(features.toDF("node", "feature2"), "node")
        .rdd
        .map(row => (row.getLong(0),
          row.getString(1) + " " + row.getString(2).replace(',', ' ')))
      // Delimiter of features and smoothed features should both be a space instead of a comma

      featureList = spark.createDataFrame(rdd_feature_merged).toDF("node", "feature")
    }

    GraphIO.save(featureList, predictOutputPath, seq = "\t")
    // The delimiter should be consistent with the input features

    PSContext.stop()
    sc.stop()
  }
}
