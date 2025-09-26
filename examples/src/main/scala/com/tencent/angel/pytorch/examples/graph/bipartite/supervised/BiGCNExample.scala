package com.tencent.angel.pytorch.examples.graph.bipartite.supervised

import com.tencent.angel.pytorch.graph.gcn.BiGCN
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.ModelGenUtils.getGNNTorchModelPathAndConfig
import com.tencent.angel.pytorch.utils.YamlParserUtils.getFeatureAndLabelPaths
import com.tencent.angel.pytorch.utils._
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.{Delimiter, GraphIO}


object BiGCNExample {

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
    val (userFeatureInput, itemFeatureInput, labelPath, testLabelPath, userFeatureSep, itemFeatureSep) = getFeatureAndLabelPaths(config)
    val nodeEmbeddingPath = FileUtils.parseDatePartitionName(config.predictor.get(Constants.NODE_EMBEDDING_OUTPUT_PATH).toString)
    val outputModelPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.MODEL_SAVE_PATH).toString)
    val featureEmbedInputPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.FEAT_EMBEDDING_LOAD_PATH).toString)

    val userFieldNum = config.model.get(Constants.INPUT_USER_FIELD_NUM).toString.toInt
    val itemFieldNum = config.model.get(Constants.INPUT_ITEM_FIELD_NUM).toString.toInt
    val edgeFieldNum = config.model.get(Constants.INPUT_EDGE_FIELD_NUM).toString.toInt
    val userFeatEmbedDim = config.model.get(Constants.INPUT_USER_EMBEDDING_DIM).toString.toInt
    val itemFeatEmbedDim = config.model.get(Constants.INPUT_ITEM_EMBEDDING_DIM).toString.toInt
    val fieldMultiHot = config.trainer.get(Constants.MULTI_HOT_FIELD).toString.toBoolean
    val batchSize = config.trainer.get(Constants.BATCH_SIZE).toString.toInt
    val stepSize = config.trainer.get(Constants.LEARNING_RATE).toString.toDouble
    val testRatio = config.trainer.get(Constants.TEST_RATIO).toString.toFloat
    val userFeatureDim = config.model.get(Constants.INPUT_USER_DIM).toString.toInt
    val itemFeatureDim = config.model.get(Constants.INPUT_ITEM_DIM).toString.toInt
    val optimizer = config.trainer.get(Constants.OPTIMIZER).toString
    val numEpoch = config.trainer.get(Constants.EPOCH).toString.toInt
    val format = config.graph.nodes.get(0).getFeature.get(Constants.FORMAT).toString
    val userNumSamples = config.trainer.get(Constants.USER_SAMPLE_NUM).toString.toInt
    val itemNumSamples = config.trainer.get(Constants.ITEM_SAMPLE_NUM).toString.toInt
    val numBatchInit = config.trainer.get(Constants.BATCH_INIT_BUM).toString.toInt
    val useSecondOrder = config.model.get(Constants.SECOND_ORDER).toString.toBoolean
    val periods = config.trainer.get(Constants.PERIODS).toString.toInt
    val checkpointInterval = config.trainer.get(Constants.CHECKPOINT_INTERVAL).toString.toInt
    val decay = config.trainer.get(Constants.DECAY).toString.toDouble
    var evals = config.trainer.get(Constants.EVAL_METRICS).toString
    val validatePeriods = config.trainer.get(Constants.VALIDATE_PERIODS).toString.toInt
    val saveCheckpoint = config.trainer.get(Constants.SAVE_CHECKPOINT).toString.toBoolean
    val hasDstType = config.model.get(Constants.ITEM_TYPES).toString.toInt > 0
    val hasEdgeType = config.model.get(Constants.EDGE_TYPES).toString.toInt > 0
    val batchSizeMultiple = config.predictor.get(Constants.BATCH_SIZE_MULTIPLIER).toString.toInt
    val isWeighted = config.graph.edges.base_config.get(Constants.WEIGHTED).toString.toBoolean
    val useSharedSamples = if (batchSize <= 128 || format.equals("sparse")) false else config.trainer.get(Constants.USE_SHARED_SAMPLES).toString.toBoolean
    val numLabels = config.trainer.get(Constants.LABELS_NUM).toString.toInt // a multi-label classification task if numLabels > 1
    val sep = Delimiter.parse(config.graph.edges.base_config.get(Constants.DELIMITER).toString)

    if (numLabels > 1) {
      evals = evals match {
        case "acc" => "multi_acc"
        case "auc" => "multi_auc"
        case _ => evals
      }
    }

    val conf = EntryUtils.start(mode, torchModelPath, "SemiBipartiteGraphsage", actionType)
    val nodeEmbeddingSep = conf.get("spark.hadoop.angel.gnn.node.embedding.value.sep", "")
    dataPartitionNum = PartitionUtils.getDataPartitionNum(dataPartitionNum, conf, dataPartitionNumFactor)
    psPartitionNum = PartitionUtils.getPsPartitionNum(psPartitionNum, conf, psPartitionNumFactor, featureEmbedInputPath)
    println(s"dataPartitionNum: $dataPartitionNum, psPartitionNum: $psPartitionNum")

    val bisage = new BiGCN()
    bisage.setTorchModelPath(torchModelPath)
    bisage.setUserFeatureDim(userFeatureDim)
    bisage.setItemFeatureDim(itemFeatureDim)
    bisage.setOptimizer(optimizer)
    bisage.setBatchSize(batchSize)
    bisage.setStepSize(stepSize)
    bisage.setPSPartitionNum(psPartitionNum)
    bisage.setPartitionNum(dataPartitionNum)
    bisage.setUseBalancePartition(useBalancePartition)
    bisage.setNumEpoch(numEpoch)
    bisage.setStorageLevel(storageLevel)
    bisage.setDataFormat(format)
    bisage.setTestRatio(testRatio)
    bisage.setUserNumSamples(userNumSamples)
    bisage.setItemNumSamples(itemNumSamples)
    bisage.setNumBatchInit(numBatchInit)
    bisage.setPeriods(periods)
    bisage.setCheckpointInterval(checkpointInterval)
    bisage.setDecay(decay)
    bisage.setEvaluations(evals)
    bisage.setValidatePeriods(validatePeriods)
    bisage.setUseSecondOrder(useSecondOrder)
    bisage.setSaveCheckpoint(saveCheckpoint)
    bisage.setHasNodeType(hasDstType)
    bisage.setHasEdgeType(hasEdgeType)
    bisage.setUseSharedSamples(useSharedSamples)
    bisage.setNumLabels(numLabels)
    bisage.setBatchSizeMultiple(batchSizeMultiple)
    bisage.setFeatEmbedPath(featureEmbedInputPath)
    bisage.setUserFeatEmbedDim(userFeatEmbedDim)
    bisage.setUserFieldNum(userFieldNum)
    bisage.setItemFeatEmbedDim(itemFeatEmbedDim)
    bisage.setItemFieldNum(itemFieldNum)
    bisage.setFieldMultiHot(fieldMultiHot)
    bisage.setHasWeighted(isWeighted)

    val edges = IOFunctions.loadEdgeFeature(edgeInput, sep = sep)

    val userFeatures = IOFunctions.loadFeature(userFeatureInput, sep = Delimiter.parse(userFeatureSep))
    val itemFeatures =  if (itemFeatureDim > 0) IOFunctions.loadFeature(itemFeatureInput, sep = Delimiter.parse(itemFeatureSep)) else null
    val labels = if (labelPath.nonEmpty) {
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath))
    } else None
    val testLabels = if (testLabelPath.nonEmpty)
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(testLabelPath, sep = "p") else IOFunctions.loadLabel(testLabelPath))
    else None

    val (model, userGraph, itemGraph) = bisage.initialize(edges, userFeatures, itemFeatures, labels, testLabels)
    bisage.showSummary(model, userGraph, itemGraph)
    if (Constants.TRAIN.equals(actionType))
      bisage.fit(model, userGraph, itemGraph, outputModelPath)

    if (nodeEmbeddingPath.nonEmpty) {
      val srcName = config.graph.edges.getLink_config.get(0).get(Constants.SRC)
      val userNodeEmbeddingOutputPath = if (nodeEmbeddingPath.trim.startsWith("tdw://")) {
        nodeEmbeddingPath + "_" + srcName
      } else {
        nodeEmbeddingPath + "/" + srcName
      }
      println(s"${srcName} embedding output path: ${userNodeEmbeddingOutputPath}")

      val userEmbedding = bisage.genLabelsEmbedding(model, userGraph)

      if (nodeEmbeddingSep.nonEmpty) {
        IOFunctions.saveEmbeddingByDelimiter(userEmbedding, nodeEmbeddingSep, userNodeEmbeddingOutputPath)
      } else {
        GraphIO.save(userEmbedding, userNodeEmbeddingOutputPath, seq = Delimiter.SPACE_VAL)
      }
    }

    if (Constants.TRAIN.equals(actionType) && outputModelPath.nonEmpty) {
      bisage.save(model, outputModelPath)
      if (userFieldNum > 0 || itemFieldNum > 0 || edgeFieldNum > 0) {
        bisage.saveFeatEmbed(model, outputModelPath)
      }
    }
    EntryUtils.stop()
  }
}
