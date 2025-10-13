package com.tencent.angel.pytorch.examples.graph.homogeneous.supervised

import com.tencent.angel.pytorch.graph.gcn.EdgePropGCN
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.ModelGenUtils.getGNNTorchModelPathAndConfig
import com.tencent.angel.pytorch.utils._
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.{Delimiter, GraphIO}


object EdgePropGCNExample {

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
    val featureInput = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getFeature.get(Constants.PATH).toString)
    val featureSep = Delimiter.parse(config.graph.nodes.get(0).feature.get(Constants.DELIMITER).toString)
    val edgeFeatureInput = FileUtils.parseDatePartitionName(config.graph.getEdges.link_config.get(0).get(Constants.PATH).toString)
    val labelPath = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getLabel.get(Constants.PATH).toString)
    val testLabelPath = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getLabel.get(Constants.VALIDATE_PATH).toString)
    val predictOutputPath = FileUtils.parseDatePartitionName(config.predictor.get(Constants.PREDICT_OUTPUT_PATH).toString)
    val outputModelPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.MODEL_SAVE_PATH).toString)
    val featureEmbedInputPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.FEAT_EMBEDDING_LOAD_PATH).toString)

    val fieldNum = config.model.get(Constants.INPUT_FIELD_NUM).toString.toInt
    val edgeFieldNum = config.model.get(Constants.INPUT_EDGE_FIELD_NUM).toString.toInt
    val featEmbedDim = config.model.get(Constants.INPUT_EMBEDDING_DIM).toString.toInt
    val fieldMultiHot = config.trainer.get(Constants.MULTI_HOT_FIELD).toString.toBoolean
    val batchSize = config.trainer.get(Constants.BATCH_SIZE).toString.toInt
    val stepSize = config.trainer.get(Constants.LEARNING_RATE).toString.toDouble
    val featureDim = config.model.get(Constants.INPUT_DIM).toString.toInt
    val edgeFeatureDim = config.model.get(Constants.INPUT_EDGE_DIM).toString.toInt
    val optimizer = config.trainer.get(Constants.OPTIMIZER).toString
    val numEpoch = config.trainer.get(Constants.EPOCH).toString.toInt
    val testRatio = config.trainer.get(Constants.TEST_RATIO).toString.toFloat
    val format = config.graph.nodes.get(0).getFeature.get(Constants.FORMAT).toString
    val numSamples = config.trainer.get(Constants.SAMPLE_NUM).toString.toInt
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
    val sep = Delimiter.parse(config.graph.edges.base_config.get(Constants.DELIMITER).toString)

    if (numLabels > 1) {
      evals = evals match {
        case "acc" => "multi_acc"
        case "auc" => "multi_auc"
        case _ => evals
      }
    }

    val conf = EntryUtils.start(mode, torchModelPath, "EdgeProp GCN", actionType)
    dataPartitionNum = PartitionUtils.getDataPartitionNum(dataPartitionNum, conf, dataPartitionNumFactor)
    psPartitionNum = PartitionUtils.getPsPartitionNum(psPartitionNum, conf, psPartitionNumFactor, featureEmbedInputPath)
    println(s"dataPartitionNum: $dataPartitionNum, psPartitionNum: $psPartitionNum")

    val gcn = new EdgePropGCN()
    gcn.setTorchModelPath(torchModelPath)
    gcn.setFeatureDim(featureDim)
    gcn.setEdgeFeatureDim(edgeFeatureDim)
    gcn.setOptimizer(optimizer)
    gcn.setBatchSize(batchSize)
    gcn.setStepSize(stepSize)
    gcn.setPSPartitionNum(psPartitionNum)
    gcn.setPartitionNum(dataPartitionNum)
    gcn.setUseBalancePartition(useBalancePartition)
    gcn.setNumEpoch(numEpoch)
    gcn.setStorageLevel(storageLevel)
    gcn.setTestRatio(testRatio)
    gcn.setDataFormat(format)
    gcn.setNumSamples(numSamples)
    gcn.setNumBatchInit(numBatchInit)
    gcn.setCheckpointInterval(checkpointInterval)
    gcn.setPeriods(periods)
    gcn.setDecay(decay)
    gcn.setEvaluations(evals)
    gcn.setValidatePeriods(validatePeriods)
    gcn.setUseSecondOrder(useSecondOrder)
    gcn.setSaveCheckpoint(saveCheckpoint)
    gcn.setUseSharedSamples(useSharedSamples)
    gcn.setNumLabels(numLabels)
    gcn.setBatchSizeMultiple(batchSizeMultiple)
    gcn.setFeatEmbedPath(featureEmbedInputPath)
    gcn.setFeatEmbedDim(featEmbedDim)
    gcn.setFieldNum(fieldNum)
    gcn.setFieldMultiHot(fieldMultiHot)

    val features = IOFunctions.loadFeature(featureInput, sep = sep)
    val edgeFeatures = IOFunctions.loadEdgeFeature(edgeFeatureInput, sep = featureSep)
    val labels = if (labelPath.nonEmpty) {
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath))
    } else None
    val testLabels = if (testLabelPath.nonEmpty)
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(testLabelPath, sep = "p") else IOFunctions.loadLabel(testLabelPath))
    else None

    val (model, graph) = gcn.initialize(edgeFeatures, features, labels, testLabels)
    gcn.showSummary(model, graph)

    if (Constants.TRAIN.equals(actionType))
      gcn.fit(model, graph, outputModelPath)

    val st = System.currentTimeMillis()
    if (predictOutputPath.nonEmpty) {
      val embedPred = gcn.genLabelsEmbedding(model, graph)
      GraphIO.save(embedPred, predictOutputPath, seq = Delimiter.SPACE_VAL)
      println(s"save predict cost: ${(System.currentTimeMillis() - st)/1000}s")
    }

    if (Constants.TRAIN.equals(actionType) && outputModelPath.nonEmpty) {
      gcn.save(model, outputModelPath)
      if (fieldNum > 0 || edgeFieldNum > 0) {
        gcn.saveFeatEmbed(model, outputModelPath)
      }
    }

    EntryUtils.stop()
  }
}
