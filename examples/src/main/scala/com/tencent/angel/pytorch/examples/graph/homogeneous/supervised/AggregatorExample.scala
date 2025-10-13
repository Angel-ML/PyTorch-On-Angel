package com.tencent.angel.pytorch.examples.graph.homogeneous.supervised

import com.tencent.angel.pytorch.graph.gcn.{Aggregator, GCN}
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.ModelGenUtils.getGNNTorchModelPathAndConfig
import com.tencent.angel.pytorch.utils._
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.{Delimiter, GraphIO}


object AggregatorExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    var psPartitionNum = params.getOrElse("psPartitionNum", "50").toInt
    var dataPartitionNum = params.getOrElse("dataPartitionNum", "100").toInt
    val psPartitionNumFactor = params.getOrElse("psPartitionNumFactor", "2").toInt
    val dataPartitionNumFactor = params.getOrElse("dataPartitionNumFactor", "3").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY").toUpperCase

    val (torchModelPath, config) = getGNNTorchModelPathAndConfig(Constants.GRAPH_TYPE_HOMOGENEOUS, Constants.TASK_TYPE_SUPERVISED)

    // path params
    val edgeInput = FileUtils.parseDatePartitionName(config.graph.getEdges.link_config.get(0).get(Constants.PATH).toString)
    val featureInput = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getFeature.get(Constants.PATH).toString)
    val predictOutputPath = FileUtils.parseDatePartitionName(config.predictor.get(Constants.PREDICT_OUTPUT_PATH).toString)
    val featureEmbedInputPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.FEAT_EMBEDDING_LOAD_PATH).toString)

    val fieldNum = config.model.get(Constants.INPUT_FIELD_NUM).toString.toInt
    val featEmbedDim = config.model.get(Constants.INPUT_EMBEDDING_DIM).toString.toInt
    val fieldMultiHot = config.trainer.get(Constants.MULTI_HOT_FIELD).toString.toBoolean
    val batchSize = config.trainer.get(Constants.BATCH_SIZE).toString.toInt
    val featureDim = config.model.get(Constants.INPUT_DIM).toString.toInt
    val format = config.graph.nodes.get(0).getFeature.get(Constants.FORMAT).toString
    val numSamples = config.trainer.get(Constants.SAMPLE_NUM).toString.toInt
    val numBatchInit = config.trainer.get(Constants.BATCH_INIT_BUM).toString.toInt
    val useSecondOrder = config.model.get(Constants.SECOND_ORDER).toString.toBoolean
    val saveCheckpoint = config.trainer.get(Constants.SAVE_CHECKPOINT).toString.toBoolean
    val checkpointInterval = config.trainer.get(Constants.CHECKPOINT_INTERVAL).toString.toInt
    val useSharedSamples = if (batchSize <= 128) false else config.trainer.get(Constants.USE_SHARED_SAMPLES).toString.toBoolean
    val batchSizeMultiple = config.predictor.get(Constants.BATCH_SIZE_MULTIPLIER).toString.toInt
    val sep = Delimiter.parse(config.graph.getEdges.base_config.get(Constants.DELIMITER).toString)
    val hops = config.trainer.get(Constants.HOPS).toString.toInt
    val featureSep = Delimiter.parse(config.graph.nodes.get(0).feature.get(Constants.DELIMITER).toString)

    val conf = EntryUtils.start(mode, torchModelPath, "Aggregator")
    dataPartitionNum = PartitionUtils.getDataPartitionNum(dataPartitionNum, conf, dataPartitionNumFactor)
    psPartitionNum = PartitionUtils.getPsPartitionNum(psPartitionNum, conf, psPartitionNumFactor, featureEmbedInputPath)
    println(s"dataPartitionNum: $dataPartitionNum, psPartitionNum: $psPartitionNum")

    /* Indeed, gcn is the aggregator here,
    but we obtain the aggregated/smoothed features with the help of
    method "genEmbedding" of class "GCN" */
    val gcn = new Aggregator()
    gcn.setTorchModelPath(torchModelPath)
    gcn.setFeatureDim(featureDim)
    gcn.setUseBalancePartition(false)
    gcn.setBatchSize(batchSize)
    gcn.setPSPartitionNum(psPartitionNum)
    gcn.setPartitionNum(dataPartitionNum)
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
    val features = IOFunctions.loadFeature(featureInput, sep = featureSep)

    val (model, graph) = gcn.initialize(edges, features, None)

    assert(predictOutputPath.nonEmpty)

    val feats = gcn.getConcatFeatures(model, graph, edges, features, hops)
    GraphIO.save(feats, predictOutputPath)

    EntryUtils.stop()
  }
}
