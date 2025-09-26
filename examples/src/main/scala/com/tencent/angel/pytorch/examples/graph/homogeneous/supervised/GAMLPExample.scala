package com.tencent.angel.pytorch.examples.graph.homogeneous.supervised

import com.tencent.angel.pytorch.graph.gcn.GAMLP
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.ModelGenUtils.getGNNTorchModelPathAndConfig
import com.tencent.angel.pytorch.utils.{Constants, EntryUtils, FileUtils, PartitionUtils}
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.{Delimiter, GraphIO}

object GAMLPExample {

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

    val (torchModelPath, config) = getGNNTorchModelPathAndConfig(Constants.GRAPH_TYPE_HOMOGENEOUS, Constants.TASK_TYPE_SUPERVISED)

    // path params
    val edgeInput = FileUtils.parseDatePartitionName(config.graph.getEdges.link_config.get(0).get(Constants.PATH).toString)
    val featureInput = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getFeature.get(Constants.PATH).toString)
    val shuffleSamples = config.trainer.get(Constants.SHUFFLE_SAMPLES).toString.toBoolean
    val labelPath = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getLabel.get(Constants.PATH).toString)
    val testLabelPath = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getLabel.get(Constants.VALIDATE_PATH).toString)
    val predictOutputPath = FileUtils.parseDatePartitionName(config.predictor.get(Constants.PREDICT_OUTPUT_PATH).toString)
    val outputModelPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.MODEL_SAVE_PATH).toString)
    val featureEmbedInputPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.FEAT_EMBEDDING_LOAD_PATH).toString)

    val fieldNum = config.model.get(Constants.INPUT_FIELD_NUM).toString.toInt
    val featEmbedDim = config.model.get(Constants.INPUT_EMBEDDING_DIM).toString.toInt
    val fieldMultiHot = config.trainer.get(Constants.MULTI_HOT_FIELD).toString.toBoolean
    val batchSize = config.trainer.get(Constants.BATCH_SIZE).toString.toInt
    val stepSize = config.trainer.get(Constants.LEARNING_RATE).toString.toDouble
    val featureDim = config.model.get(Constants.INPUT_DIM).toString.toInt
    val optimizer = config.trainer.get(Constants.OPTIMIZER).toString
    val numEpoch = config.trainer.get(Constants.EPOCH).toString.toInt
    val testRatio = config.trainer.get(Constants.TEST_RATIO).toString.toFloat
    val format = config.graph.nodes.get(0).getFeature.get(Constants.FORMAT).toString
    val numBatchInit = config.trainer.get(Constants.BATCH_INIT_BUM).toString.toInt
    val periods = config.trainer.get(Constants.PERIODS).toString.toInt
    val decay = config.trainer.get(Constants.DECAY).toString.toDouble
    val validatePeriods = config.trainer.get(Constants.VALIDATE_PERIODS).toString.toInt
    val useSecondOrder = config.model.get(Constants.SECOND_ORDER).toString.toBoolean
    val saveCheckpoint = config.trainer.get(Constants.SAVE_CHECKPOINT).toString.toBoolean
    val checkpointInterval = config.trainer.get(Constants.CHECKPOINT_INTERVAL).toString.toInt
    val useSharedSamples = if (batchSize <= 128) false else config.trainer.get(Constants.USE_SHARED_SAMPLES).toString.toBoolean
    val numLabels = config.trainer.get(Constants.LABELS_NUM).toString.toInt // a multi-label classification task if numLabels > 1
    var evals = config.trainer.get(Constants.EVAL_METRICS).toString
    val batchSizeMultiple = config.predictor.get(Constants.BATCH_SIZE_MULTIPLIER).toString.toInt
    val sep = Delimiter.parse(config.graph.getEdges.base_config.get(Constants.DELIMITER).toString)
    val featureSep = Delimiter.parse(config.graph.nodes.get(0).feature.get(Constants.DELIMITER).toString)
    val labelSep = Delimiter.parse(config.graph.getNodes.get(0).getLabel.get(Constants.DELIMITER).toString)
    val hops = config.model.get(Constants.HOPS).toString.toInt

    if (numLabels > 1) evals = "multi_auc"

    val conf = EntryUtils.start(mode, torchModelPath, "GAMLP", actionType)
    dataPartitionNum = PartitionUtils.getDataPartitionNum(dataPartitionNum, conf, dataPartitionNumFactor)
    psPartitionNum = PartitionUtils.getPsPartitionNum(psPartitionNum, conf, psPartitionNumFactor, featureEmbedInputPath)
    println(s"dataPartitionNum: $dataPartitionNum, psPartitionNum: $psPartitionNum")


    val gamlp = new GAMLP()
    gamlp.setTorchModelPath(torchModelPath)
    gamlp.setFeatureDim(featureDim * (hops + 1))
    gamlp.setOptimizer(optimizer)
    gamlp.setBatchSize(batchSize)
    gamlp.setStepSize(stepSize)
    gamlp.setPSPartitionNum(psPartitionNum)
    gamlp.setPartitionNum(dataPartitionNum)
    gamlp.setUseBalancePartition(useBalancePartition)
    gamlp.setNumEpoch(numEpoch)
    gamlp.setStorageLevel(storageLevel)
    gamlp.setTestRatio(testRatio)
    gamlp.setDataFormat(format)
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
    val features = IOFunctions.loadFeature(featureInput, sep = featureSep)
    val labels = if (labelPath.length > 0) {
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath, sep=labelSep))
    } else None
    val testLabels = if (testLabelPath.length > 0)
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(testLabelPath, sep = "p") else IOFunctions.loadLabel(testLabelPath, sep=labelSep))
    else None

    val (model, graph) = gamlp.initialize(edges, features, labels, testLabels)
    gamlp.showSummary(model, graph)

    if (Constants.TRAIN.equals(actionType))
      gamlp.fit(model, graph, outputModelPath)

    if (predictOutputPath.nonEmpty) {
      val predict = gamlp.genLabels(model, graph)
      GraphIO.save(predict, predictOutputPath, seq = Delimiter.SPACE_VAL)
    }

    if (Constants.TRAIN.equals(actionType) && outputModelPath.nonEmpty) {
      gamlp.save(model, outputModelPath)
      if (fieldNum > 0) {
        gamlp.saveFeatEmbed(model, outputModelPath)
      }
    }
    EntryUtils.stop()
  }
}
