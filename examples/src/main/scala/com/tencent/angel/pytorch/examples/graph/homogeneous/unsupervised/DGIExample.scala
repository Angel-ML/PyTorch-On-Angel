package com.tencent.angel.pytorch.examples.graph.homogeneous.unsupervised

import com.tencent.angel.pytorch.graph.gcn.DGI
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.ModelGenUtils.getGNNTorchModelPathAndConfig
import com.tencent.angel.pytorch.utils._
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.{Delimiter, GraphIO}

import scala.language.existentials

object DGIExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    var psPartitionNum = params.getOrElse("psPartitionNum", "50").toInt
    var dataPartitionNum = params.getOrElse("dataPartitionNum", "100").toInt
    val psPartitionNumFactor = params.getOrElse("psPartitionNumFactor", "2").toInt
    val dataPartitionNumFactor = params.getOrElse("dataPartitionNumFactor", "3").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY").toUpperCase
    val actionType = params.getOrElse("actionType", "train")

    val (torchModelPath, config) = getGNNTorchModelPathAndConfig(Constants.GRAPH_TYPE_HOMOGENEOUS, Constants.TASK_TYPE_UNSUPERVISED)

    val edgeInput = FileUtils.parseDatePartitionName(config.graph.edges.link_config.get(0).get(Constants.PATH).toString)
    val featureInput = FileUtils.parseDatePartitionName(config.graph.nodes.get(0).feature.get(Constants.PATH).toString)
    val featureSep = config.graph.nodes.get(0).feature.get(Constants.DELIMITER).toString
    val outputModelPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.MODEL_SAVE_PATH).toString)
    val featEmbeddingLoadPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.FEAT_EMBEDDING_LOAD_PATH).toString)
    val nodeEmbeddingOutputPath = FileUtils.parseDatePartitionName(config.predictor.get(Constants.NODE_EMBEDDING_OUTPUT_PATH).toString)

    val isWeighted = config.graph.getEdges.base_config.get(Constants.WEIGHTED).toString.toBoolean
    val sep = Delimiter.parse(config.graph.edges.base_config.get(Constants.DELIMITER).toString)
    val multihotField = config.trainer.get(Constants.MULTI_HOT_FIELD).toString.toBoolean
    val batchSize = config.trainer.get(Constants.BATCH_SIZE).toString.toInt
    val stepSize = config.trainer.get(Constants.LEARNING_RATE).toString.toDouble
    val decay = config.trainer.get(Constants.DECAY).toString.toDouble
    val optimizer = config.trainer.get(Constants.OPTIMIZER).toString
    val epoch = config.trainer.get(Constants.EPOCH).toString.toInt
    val second = config.model.get(Constants.SECOND_ORDER).toString.toBoolean
    val format = config.graph.nodes.get(0).feature.get(Constants.FORMAT).toString
    val sampleNum = config.trainer.get(Constants.SAMPLE_NUM).toString.toInt
    val batchInitNum = config.trainer.get(Constants.BATCH_INIT_BUM).toString.toInt
    val saveCheckpoint = config.trainer.get(Constants.SAVE_CHECKPOINT).toString.toBoolean
    val periods = config.trainer.get(Constants.PERIODS).toString.toInt
    val batchSizeMultiplier = config.predictor.get(Constants.BATCH_SIZE_MULTIPLIER).toString.toInt
    //random sample trainRatio samples to train in each epoch
    val trainSampleRatio = config.trainer.get(Constants.TRAIN_SAMPLE_RATIO).toString.toFloat
    val featureDim = config.model.get(Constants.INPUT_DIM).toString.toInt
    val fieldNum = config.model.get(Constants.INPUT_FIELD_NUM).toString.toInt
    val featEmbedDim = config.model.get(Constants.INPUT_EMBEDDING_DIM).toString.toInt
    val conf = EntryUtils.start(mode, torchModelPath, "DGIExample", actionType)
    // sep between embedding value
    val nodeEmbeddingSep = conf.get("spark.hadoop.angel.gnn.node.embedding.value.sep", "")
    dataPartitionNum = PartitionUtils.getDataPartitionNum(dataPartitionNum, conf, dataPartitionNumFactor)
    psPartitionNum = PartitionUtils.getPsPartitionNum(psPartitionNum, conf, psPartitionNumFactor, featEmbeddingLoadPath)
    println(s"dataPartitionNum: $dataPartitionNum, psPartitionNum: $psPartitionNum")

    val dgi = new DGI()
    dgi.setTorchModelPath(torchModelPath)
    dgi.setFeatureDim(featureDim)
    dgi.setOptimizer(optimizer)
    dgi.setBatchSize(batchSize)
    dgi.setStepSize(stepSize)
    dgi.setPSPartitionNum(psPartitionNum)
    dgi.setPartitionNum(dataPartitionNum)
    dgi.setNumEpoch(epoch)
    dgi.setStorageLevel(storageLevel)
    dgi.setUseSecondOrder(second)
    dgi.setDataFormat(format)
    dgi.setNumBatchInit(batchInitNum)
    dgi.setNumSamples(sampleNum)
    dgi.setSaveCheckpoint(saveCheckpoint)
    dgi.setPeriods(periods)
    dgi.setDecay(decay)
    dgi.setBatchSizeMultiple(batchSizeMultiplier)
    dgi.setFeatEmbedPath(featEmbeddingLoadPath)
    dgi.setFeatEmbedDim(featEmbedDim)
    dgi.setFieldNum(fieldNum)
    dgi.setTestRatio(trainSampleRatio)
    dgi.setFieldMultiHot(multihotField)
    dgi.setHasWeighted(isWeighted)

    val edges = GraphIO.load(edgeInput, isWeighted = isWeighted, sep = sep)
    val features = IOFunctions.loadFeature(featureInput, sep = Delimiter.parse(featureSep))
    val (model, graph) = dgi.initialize(edges, features)
    dgi.showSummary(model, graph)

    if (Constants.TRAIN.equals(actionType))
      dgi.fit(model, graph, outputModelPath)

    val start = System.currentTimeMillis()
    if (nodeEmbeddingOutputPath.nonEmpty) {
      val embedding = dgi.genEmbedding(model, graph)
      if (nodeEmbeddingSep.nonEmpty) {
        IOFunctions.saveEmbeddingByDelimiter(embedding, nodeEmbeddingSep, nodeEmbeddingOutputPath)
      } else {
        GraphIO.save(embedding, nodeEmbeddingOutputPath, seq = Delimiter.SPACE_VAL)
      }
      println(s"save embedding cost: ${(System.currentTimeMillis() - start)/1000}s")
    }

    if (Constants.TRAIN.equals(actionType) && outputModelPath.nonEmpty) {
      dgi.save(model, outputModelPath)
      if (fieldNum > 0) {
        dgi.saveFeatEmbed(model, outputModelPath)
      }
    }

    EntryUtils.stop()
  }
}
