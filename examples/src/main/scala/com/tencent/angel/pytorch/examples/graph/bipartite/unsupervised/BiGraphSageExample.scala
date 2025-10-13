package com.tencent.angel.pytorch.examples.graph.bipartite.unsupervised

import com.tencent.angel.pytorch.graph.gcn.BiSAGE
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.ModelGenUtils.getGNNTorchModelPathAndConfig
import com.tencent.angel.pytorch.utils.YamlParserUtils.getFeatureAndQueryPaths
import com.tencent.angel.pytorch.utils._
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.{Delimiter, GraphIO}

import scala.language.existentials

object BiGraphSageExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    var psPartitionNum = params.getOrElse("psPartitionNum", "50").toInt
    var dataPartitionNum = params.getOrElse("dataPartitionNum", "100").toInt
    val psPartitionNumFactor = params.getOrElse("psPartitionNumFactor", "2").toInt
    val dataPartitionNumFactor = params.getOrElse("dataPartitionNumFactor", "3").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY").toUpperCase
    val actionType = params.getOrElse("actionType", "train")

    val (torchModelPath, config) = getGNNTorchModelPathAndConfig(Constants.GRAPH_TYPE_BIPARTITE, Constants.TASK_TYPE_UNSUPERVISED)

    val edgeInput = FileUtils.parseDatePartitionName(config.graph.edges.link_config.get(0).get(Constants.PATH).toString)
    val (userFeatureInput, itemFeatureInput, userQueryPath, itemQueryPath, userFeatureSep, itemFeatureSep) = getFeatureAndQueryPaths(config)
    val nodeEmbeddingOutputPath = FileUtils.parseDatePartitionName(config.predictor.get(Constants.NODE_EMBEDDING_OUTPUT_PATH).toString)
    val outputModelPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.MODEL_SAVE_PATH).toString)
    val featureEmbedInputPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.FEAT_EMBEDDING_LOAD_PATH).toString)

    val userFieldNum = config.model.get(Constants.INPUT_USER_FIELD_NUM).toString.toInt
    val itemFieldNum = config.model.get(Constants.INPUT_ITEM_FIELD_NUM).toString.toInt
    val userFeatEmbedDim = config.model.get(Constants.INPUT_USER_EMBEDDING_DIM).toString.toInt
    val itemFeatEmbedDim = config.model.get(Constants.INPUT_ITEM_EMBEDDING_DIM).toString.toInt
    val fieldMultiHot = config.trainer.get(Constants.MULTI_HOT_FIELD).toString.toBoolean
    val batchSize = config.trainer.get(Constants.BATCH_SIZE).toString.toInt
    val stepSize = config.trainer.get(Constants.LEARNING_RATE).toString.toDouble
    val decay = config.trainer.get(Constants.DECAY).toString.toDouble
    val userFeatureDim = config.model.get(Constants.INPUT_USER_DIM).toString.toInt
    val itemFeatureDim = config.model.get(Constants.INPUT_ITEM_DIM).toString.toInt
    val optimizer = config.trainer.get(Constants.OPTIMIZER).toString
    val numEpoch = config.trainer.get(Constants.EPOCH).toString.toInt
    val second = config.model.get(Constants.SECOND_ORDER).toString.toBoolean
    val format = config.graph.nodes.get(0).getFeature.get(Constants.FORMAT).toString
    val periods = config.trainer.get(Constants.PERIODS).toString.toInt
    val batchSizeMultiple = config.predictor.get(Constants.BATCH_SIZE_MULTIPLIER).toString.toInt
    val userNumSamples = config.trainer.get(Constants.USER_SAMPLE_NUM).toString.toInt
    val itemNumSamples = config.trainer.get(Constants.ITEM_SAMPLE_NUM).toString.toInt
    val trainSampleRatio = config.trainer.get(Constants.TRAIN_SAMPLE_RATIO).toString.toFloat
    val saveCheckpoint = config.trainer.get(Constants.SAVE_CHECKPOINT).toString.toBoolean
    val sep = Delimiter.parse(config.graph.edges.base_config.get(Constants.DELIMITER).toString)

    val conf = EntryUtils.start(mode, torchModelPath, "UnSupervisedBiGraphSageExample", actionType)
    // sep between embedding value
    val nodeEmbeddingSep = conf.get("spark.hadoop.angel.gnn.node.embedding.value.sep", "")
    dataPartitionNum = PartitionUtils.getDataPartitionNum(dataPartitionNum, conf, dataPartitionNumFactor)
    psPartitionNum = PartitionUtils.getPsPartitionNum(psPartitionNum, conf, psPartitionNumFactor, featureEmbedInputPath)
    println(s"dataPartitionNum: $dataPartitionNum, psPartitionNum: $psPartitionNum")

    val bisage = new BiSAGE()
    bisage.setTorchModelPath(torchModelPath)
    bisage.setUserFeatureDim(userFeatureDim)
    bisage.setItemFeatureDim(itemFeatureDim)
    bisage.setOptimizer(optimizer)
    bisage.setBatchSize(batchSize)
    bisage.setStepSize(stepSize)
    bisage.setPSPartitionNum(psPartitionNum)
    bisage.setPartitionNum(dataPartitionNum)
    bisage.setNumEpoch(numEpoch)
    bisage.setStorageLevel(storageLevel)
    bisage.setUseSecondOrder(second)
    bisage.setDataFormat(format)
    bisage.setSaveCheckpoint(saveCheckpoint)
    bisage.setPeriods(periods)
    bisage.setDecay(decay)
    bisage.setBatchSizeMultiple(batchSizeMultiple)
    bisage.setFeatEmbedPath(featureEmbedInputPath)
    bisage.setUserFeatEmbedDim(userFeatEmbedDim)
    bisage.setUserFieldNum(userFieldNum)
    bisage.setItemFeatEmbedDim(itemFeatEmbedDim)
    bisage.setItemFieldNum(itemFieldNum)
    bisage.setFieldMultiHot(fieldMultiHot)
    bisage.setUserNumSamples(userNumSamples)
    bisage.setItemNumSamples(itemNumSamples)
    bisage.setTestRatio(trainSampleRatio)

    val edges = GraphIO.load(edgeInput, isWeighted = false, sep = sep)
    val userFeatures = IOFunctions.loadFeature(userFeatureInput, sep = Delimiter.parse(userFeatureSep))
    val itemFeatures = if (itemFeatureDim > 0) IOFunctions.loadFeature(itemFeatureInput, sep = Delimiter.parse(itemFeatureSep)) else null
    val (model, userGraph, itemGraph) = bisage.initialize(edges, userFeatures, itemFeatures)

    bisage.showSummary(model, userGraph, itemGraph)
    if (Constants.TRAIN.equals(actionType))
      bisage.fit(model, userGraph, itemGraph, outputModelPath)

    // save embedding
    if (nodeEmbeddingOutputPath.nonEmpty) {
      val srcName = config.graph.edges.getLink_config.get(0).get(Constants.SRC)
      val dstName = config.graph.edges.getLink_config.get(0).get(Constants.DST)
      val userNodeEmbeddingOutputPath = if (nodeEmbeddingOutputPath.trim.startsWith("tdw://")) {
        nodeEmbeddingOutputPath + "_" + srcName
      } else {
        nodeEmbeddingOutputPath + "/" + srcName
      }
      val itemNodeEmbeddingOutputPath = if (nodeEmbeddingOutputPath.trim.startsWith("tdw://")) {
        nodeEmbeddingOutputPath + "_" + dstName
      } else {
        nodeEmbeddingOutputPath + "/" + dstName
      }
      println(s"${srcName} embedding output path: ${userNodeEmbeddingOutputPath}")
      println(s"${dstName} embedding output path: ${itemNodeEmbeddingOutputPath}")

      val userEmbedding = bisage.genEmbedding(model, userGraph, 0)
      val itemEmbedding = bisage.genEmbedding(model, itemGraph, 1)

      if (nodeEmbeddingSep.nonEmpty) {
        IOFunctions.saveEmbeddingByDelimiter(userEmbedding, nodeEmbeddingSep, userNodeEmbeddingOutputPath)
        IOFunctions.saveEmbeddingByDelimiter(itemEmbedding, nodeEmbeddingSep, itemNodeEmbeddingOutputPath)
      } else {
        GraphIO.save(userEmbedding, userNodeEmbeddingOutputPath, seq = Delimiter.SPACE_VAL)
        GraphIO.save(itemEmbedding, itemNodeEmbeddingOutputPath, seq = Delimiter.SPACE_VAL)
      }
    }

    if (Constants.TRAIN.equals(actionType) && outputModelPath.nonEmpty) {
      bisage.save(model, outputModelPath)
      if (userFieldNum > 0) {
        bisage.saveFeatEmbed(model, outputModelPath)
      }
    }

    EntryUtils.stop()
  }

}
