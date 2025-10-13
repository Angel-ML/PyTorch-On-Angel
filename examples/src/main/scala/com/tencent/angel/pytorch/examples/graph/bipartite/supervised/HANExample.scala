package com.tencent.angel.pytorch.examples.graph.bipartite.supervised

import com.tencent.angel.pytorch.graph.gcn.hetAttention.HAN
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.ModelGenUtils.getGNNTorchModelPathAndConfig
import com.tencent.angel.pytorch.utils._
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.{Delimiter, GraphIO}

object HANExample {

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

    // path params
    val edgeInput = FileUtils.parseDatePartitionName(config.graph.getEdges.link_config.get(0).get(Constants.PATH).toString)
    val featureInput = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getFeature.get(Constants.PATH).toString)
    val labelPath = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getLabel.get(Constants.PATH).toString)
    val testLabelPath = FileUtils.parseDatePartitionName(config.graph.getNodes.get(0).getLabel.get(Constants.VALIDATE_PATH).toString)
    val nodeEmbeddingPath = FileUtils.parseDatePartitionName(config.predictor.get(Constants.NODE_EMBEDDING_OUTPUT_PATH).toString)
    val outputModelPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.MODEL_SAVE_PATH).toString)
    val featureEmbedInputPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.FEAT_EMBEDDING_LOAD_PATH).toString)

    val fieldNum = config.model.get(Constants.INPUT_USER_FIELD_NUM).toString.toInt
    val featEmbedDim = config.model.get(Constants.INPUT_USER_EMBEDDING_DIM).toString.toInt
    val fieldMultiHot = config.trainer.get(Constants.MULTI_HOT_FIELD).toString.toBoolean
    val batchSize = config.trainer.get(Constants.BATCH_SIZE).toString.toInt
    val stepSize = config.trainer.get(Constants.LEARNING_RATE).toString.toDouble
    val featureDim = config.model.get(Constants.INPUT_DIM).toString.toInt
    val optimizer = config.trainer.get(Constants.OPTIMIZER).toString
    val numEpoch = config.trainer.get(Constants.EPOCH).toString.toInt
    val testRatio = config.trainer.get(Constants.TEST_RATIO).toString.toFloat
    val format = config.graph.nodes.get(0).getFeature.get(Constants.FORMAT).toString
    val numSamples = config.trainer.get(Constants.USER_SAMPLE_NUM).toString.toInt
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
    val itemTypes = config.model.get(Constants.ITEM_TYPES).toString.toInt
    val hasNodeType = itemTypes > 0
    val sep = Delimiter.parse(config.graph.getEdges.base_config.get(Constants.DELIMITER).toString)
    val userFeatureSep = Delimiter.parse(config.graph.nodes.get(0).feature.get(Constants.DELIMITER).toString)

    if (numLabels > 1) {
      evals = evals match {
        case "acc" => "multi_acc"
        case "auc" => "multi_auc"
        case _ => evals
      }
    }

    val conf = EntryUtils.start(mode, torchModelPath, "HAN", actionType)
    // sep between embedding value
    val nodeEmbeddingSep = conf.get("spark.hadoop.angel.gnn.node.embedding.value.sep", "")
    dataPartitionNum = PartitionUtils.getDataPartitionNum(dataPartitionNum, conf, dataPartitionNumFactor)
    psPartitionNum = PartitionUtils.getPsPartitionNum(psPartitionNum, conf, psPartitionNumFactor, featureEmbedInputPath)
    println(s"dataPartitionNum: $dataPartitionNum, psPartitionNum: $psPartitionNum")

    val han = new HAN()
    han.setTorchModelPath(torchModelPath)
    han.setOptimizer(optimizer)
    han.setBatchSize(batchSize)
    han.setStepSize(stepSize)
    han.setPSPartitionNum(psPartitionNum)
    han.setPartitionNum(dataPartitionNum)
    han.setUseBalancePartition(useBalancePartition)
    han.setNumEpoch(numEpoch)
    han.setStorageLevel(storageLevel)
    han.setDataFormat(format)
    han.setTestRatio(testRatio)
    han.setNumSamples(numSamples)
    han.setNumBatchInit(numBatchInit)
    han.setPeriods(periods)
    han.setCheckpointInterval(checkpointInterval)
    han.setDecay(decay)
    han.setEvaluations(evals)
    han.setValidatePeriods(validatePeriods)
    han.setUseSecondOrder(useSecondOrder)
    han.setSaveCheckpoint(saveCheckpoint)
    han.setHasNodeType(hasNodeType)
    han.setItemTypes(itemTypes)
    han.setFeatureDim(featureDim)
    han.setNumLabels(numLabels)
    han.setBatchSizeMultiple(batchSizeMultiple)
    han.setFeatEmbedPath(featureEmbedInputPath)
    han.setFeatEmbedDim(featEmbedDim)
    han.setFieldNum(fieldNum)
    han.setFieldMultiHot(fieldMultiHot)
    han.setUseSharedSamples(useSharedSamples)

    val edges = GraphIO.load(edgeInput, isWeighted = hasNodeType, sep = sep)
    val userFeatures = IOFunctions.loadFeature(featureInput, sep = userFeatureSep)
    val labels = if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath)
    val testLabels = if (testLabelPath.nonEmpty)
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(testLabelPath, sep = "p") else IOFunctions.loadLabel(testLabelPath))
    else None

    val (model, graph) = han.initialize(edges, userFeatures, Option(labels), testLabels)
    han.showSummary(model, graph)
    if (Constants.TRAIN.equals(actionType))
      han.fit(model, graph)

    if (nodeEmbeddingPath.nonEmpty) {
      val srcName = config.graph.edges.getLink_config.get(0).get(Constants.SRC)
      val userNodeEmbeddingOutputPath = if (nodeEmbeddingPath.trim.startsWith("tdw://")) {
        nodeEmbeddingPath + "_" + srcName
      } else {
        nodeEmbeddingPath + "/" + srcName
      }
      println(s"${srcName} embedding output path: ${userNodeEmbeddingOutputPath}")
      val userEmbedding = han.genLabelsEmbedding(model, graph)

      if (nodeEmbeddingSep.nonEmpty) {
        IOFunctions.saveEmbeddingByDelimiter(userEmbedding, nodeEmbeddingSep, userNodeEmbeddingOutputPath)
      } else {
        GraphIO.save(userEmbedding, userNodeEmbeddingOutputPath, seq = Delimiter.SPACE_VAL)
      }
    }

    if (Constants.TRAIN.equals(actionType) && outputModelPath.nonEmpty) {
      han.save(model, outputModelPath)
      if (fieldNum > 0) {
        han.saveFeatEmbed(model, outputModelPath)
      }
    }
    EntryUtils.stop()
  }
}
