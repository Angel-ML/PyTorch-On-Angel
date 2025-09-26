package com.tencent.angel.pytorch.examples.graph.bipartite.supervised

import com.tencent.angel.pytorch.graph.gcn.IGMC
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.ModelGenUtils.getGNNTorchModelPathAndConfig
import com.tencent.angel.pytorch.utils.YamlParserUtils.getFeatureAndQueryPaths
import com.tencent.angel.pytorch.utils._
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.{Delimiter, GraphIO}


object IGMCExample {
  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    var psPartitionNum = params.getOrElse("psPartitionNum", "50").toInt
    var dataPartitionNum = params.getOrElse("dataPartitionNum", "100").toInt
    val psPartitionNumFactor = params.getOrElse("psPartitionNumFactor", "2").toInt
    val dataPartitionNumFactor = params.getOrElse("dataPartitionNumFactor", "3").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY").toUpperCase
    val actionType = params.getOrElse("actionType", "train")

    val (torchModelPath, config) = getGNNTorchModelPathAndConfig(Constants.GRAPH_TYPE_BIPARTITE, Constants.TASK_TYPE_SUPERVISED)

    // path param
    val edgeInput = FileUtils.parseDatePartitionName(config.graph.edges.link_config.get(0).get(Constants.PATH).toString)
    val (userFeatureInput, itemFeatureInput, _, _, userFeatureSep, itemFeatureSep) = getFeatureAndQueryPaths(config)
    val predictOutputPath = FileUtils.parseDatePartitionName(config.predictor.get(Constants.PREDICT_OUTPUT_PATH).toString)
    val outputModelPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.MODEL_SAVE_PATH).toString)

    val batchSize = config.trainer.get(Constants.BATCH_SIZE).toString.toInt
    val stepSize = config.trainer.get(Constants.LEARNING_RATE).toString.toDouble
    val testRatio = config.trainer.get(Constants.TEST_RATIO).toString.toFloat
    val userFeatureDim = config.model.get(Constants.INPUT_USER_DIM).toString.toInt
    val itemFeatureDim = config.model.get(Constants.INPUT_ITEM_DIM).toString.toInt
    val optimizer = config.trainer.get(Constants.OPTIMIZER).toString
    val numEpoch = config.trainer.get(Constants.EPOCH).toString.toInt
    val format = config.graph.nodes.get(0).getFeature.get(Constants.FORMAT).toString
    val numSamples = config.trainer.get(Constants.USER_SAMPLE_NUM).toString.toInt
    val numBatchInit = config.trainer.get(Constants.BATCH_INIT_BUM).toString.toInt
    val useSecondOrder = config.model.get(Constants.SECOND_ORDER).toString.toBoolean
    val periods = config.trainer.get(Constants.PERIODS).toString.toInt
    val checkpointInterval = config.trainer.get(Constants.CHECKPOINT_INTERVAL).toString.toInt
    val decay = config.trainer.get(Constants.DECAY).toString.toDouble
    val evals = config.trainer.get(Constants.EVAL_METRICS).toString
    val validatePeriods = config.trainer.get(Constants.VALIDATE_PERIODS).toString.toInt
    val saveCheckpoint = config.trainer.get(Constants.SAVE_CHECKPOINT).toString.toBoolean
    val hasEdgeType = config.model.get(Constants.EDGE_TYPES).toString.toInt > 0
    val taskType = config.model.get(Constants.TASK_TYPE).toString
    val batchSizeMultiple = config.predictor.get(Constants.BATCH_SIZE_MULTIPLIER).toString.toInt
    val useSharedSamples = if (batchSize <= 128 || format.equals("sparse")) false else config.trainer.get(Constants.USE_SHARED_SAMPLES).toString.toBoolean
    val sep = Delimiter.parse(config.graph.edges.base_config.get(Constants.DELIMITER).toString)

    val conf = EntryUtils.start(mode, torchModelPath, "IGMC", actionType)
    dataPartitionNum = PartitionUtils.getDataPartitionNum(dataPartitionNum, conf, dataPartitionNumFactor)
    psPartitionNum = PartitionUtils.getPsPartitionNum(psPartitionNum, conf, psPartitionNumFactor)
    println(s"dataPartitionNum: $dataPartitionNum, psPartitionNum: $psPartitionNum")

    val igmc = new IGMC()
    igmc.setTorchModelPath(torchModelPath)
    igmc.setUserFeatureDim(userFeatureDim)
    igmc.setItemFeatureDim(itemFeatureDim)
    igmc.setOptimizer(optimizer)
    igmc.setBatchSize(batchSize)
    igmc.setStepSize(stepSize)
    igmc.setPSPartitionNum(psPartitionNum)
    igmc.setPartitionNum(dataPartitionNum)
    igmc.setUseBalancePartition(useBalancePartition)
    igmc.setNumEpoch(numEpoch)
    igmc.setStorageLevel(storageLevel)
    igmc.setDataFormat(format)
    igmc.setTestRatio(testRatio)
    igmc.setNumSamples(numSamples)
    igmc.setNumBatchInit(numBatchInit)
    igmc.setPeriods(periods)
    igmc.setCheckpointInterval(checkpointInterval)
    igmc.setDecay(decay)
    igmc.setEvaluations(evals)
    igmc.setValidatePeriods(validatePeriods)
    igmc.setUseSecondOrder(useSecondOrder)
    igmc.setSaveCheckpoint(saveCheckpoint)
    igmc.setHasEdgeType(hasEdgeType)
    igmc.setTaskType(taskType)
    igmc.setUseSharedSamples(useSharedSamples)
    igmc.setBatchSizeMultiple(batchSizeMultiple)

    val edges = GraphIO.load(edgeInput, isWeighted = hasEdgeType, sep = sep)
    val userFeatures = if (userFeatureDim > 0) IOFunctions.loadFeature(userFeatureInput, sep = Delimiter.parse(userFeatureSep)) else null
    val itemFeatures = if (itemFeatureDim > 0) IOFunctions.loadFeature(itemFeatureInput, sep = Delimiter.parse(itemFeatureSep)) else null

    val (model, userGraph, itemGraph) = igmc.initialize(edges, userFeatures, itemFeatures, None, None)
    igmc.showSummary(model, userGraph, itemGraph)
    if (Constants.TRAIN.equals(actionType))
      igmc.fit(model, userGraph, itemGraph, outputModelPath)

    if (predictOutputPath.nonEmpty) {
      val predicts = igmc.genLabels(model, userGraph)
      GraphIO.save(predicts, predictOutputPath, seq = Delimiter.SPACE_VAL)
    }

    if (Constants.TRAIN.equals(actionType) && outputModelPath.nonEmpty)
      igmc.save(model, outputModelPath)

    EntryUtils.stop()
  }
}
