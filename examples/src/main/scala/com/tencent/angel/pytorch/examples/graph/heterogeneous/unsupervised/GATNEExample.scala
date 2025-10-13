package com.tencent.angel.pytorch.examples.graph.heterogeneous.unsupervised

import com.tencent.angel.pytorch.graph.gcn.GATNE
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.ModelGenUtils.getGNNTorchModelPathAndConfig
import com.tencent.angel.pytorch.utils.YamlParserUtils.parseNodePaths
import com.tencent.angel.pytorch.utils.{Constants, EntryUtils, FileUtils, PartitionUtils}
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.{Delimiter, GraphIO}


object GATNEExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    var psPartitionNum = params.getOrElse("psPartitionNum", "50").toInt
    var dataPartitionNum = params.getOrElse("dataPartitionNum", "100").toInt
    val psPartitionNumFactor = params.getOrElse("psPartitionNumFactor", "2").toInt
    val dataPartitionNumFactor = params.getOrElse("dataPartitionNumFactor", "3").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_AND_DISK").toUpperCase
    val actionType = params.getOrElse("actionType", "train")

    val (torchModelPath, config) = getGNNTorchModelPathAndConfig(Constants.GRAPH_TYPE_HETEROGENEOUS, Constants.TASK_TYPE_UNSUPERVISED)

    val edgeInput = FileUtils.parseDatePartitionName(config.graph.getEdges.link_config.get(0).get(Constants.PATH).toString)
    val testEdgeInput = FileUtils.parseDatePartitionName(config.trainer.get(Constants.TEST_EDGE_PATH).toString)
    val featEmbeddingLoadPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.FEAT_EMBEDDING_LOAD_PATH).toString)
    val ContextEmbeddingInputPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.CONTEXT_EMBEDDING_LOAD_PATH).toString)
    val fieldMultiHot = config.trainer.get(Constants.MULTI_HOT_FIELD).toString.toBoolean
    val walkInput = FileUtils.parseDatePartitionName(config.trainer.get(Constants.WALK_PATH).toString)
    val nodeTypeInput = FileUtils.parseDatePartitionName(config.trainer.get(Constants.NODE_TYPE_PATH).toString)
    val embeddingOutputPath = FileUtils.parseDatePartitionName(config.predictor.get(Constants.NODE_EMBEDDING_OUTPUT_PATH).toString)
    val outputModelPath = FileUtils.parseDatePartitionName(config.trainer.get(Constants.MODEL_SAVE_PATH).toString)

    val optimizer = config.trainer.get(Constants.OPTIMIZER).toString
    val batchSize = config.trainer.get(Constants.BATCH_SIZE).toString.toInt
    val stepSize = config.trainer.get(Constants.LEARNING_RATE).toString.toDouble
    val decay = config.trainer.get(Constants.DECAY).toString.toDouble
    val epoch = config.trainer.get(Constants.EPOCH).toString.toInt
    val format = config.graph.nodes.get(0).feature.get(Constants.FORMAT).toString
    val evals = config.trainer.get(Constants.EVAL_METRICS).toString
    val sep = Delimiter.parse(config.graph.edges.base_config.get(Constants.DELIMITER).toString)
    val nodesSampleNum = config.trainer.get(Constants.NODES_SAMPLE_NUM).toString
    val contextDim = config.model.get(Constants.EMBEDDING_DIM).toString.toInt
    val windowSize = config.trainer.get(Constants.WINDOW_SIZE).toString.toInt
    val numNegSamples = config.trainer.get(Constants.NEGATIVE_SAMPLE_NUM).toString.toInt
    val iniFuncType = config.trainer.get(Constants.INIT_METHOD).toString
    val logStep = config.trainer.get(Constants.LOG_STEP).toString.toInt
    val mean = config.trainer.get(Constants.MEAN).toString.toFloat
    val std = config.trainer.get(Constants.STD).toString.toFloat
    val negSampleByNodeType = config.trainer.get(Constants.NEGATIVE_SAMPLE_BY_NODE_TYPE).toString.toBoolean
    val evaluateByEdgeType = config.trainer.get(Constants.EVALUATE_BY_EDGE_TYPE).toString.toBoolean
    val validatePeriods = config.trainer.get(Constants.VALIDATE_PERIODS).toString.toInt
    val saveCheckpoint = config.trainer.get(Constants.SAVE_CHECKPOINT).toString.toBoolean
    val saveModelInterval = config.trainer.get(Constants.SAVE_MODEL_INTERVAL).toString.toInt
    val partitionOpt = config.trainer.get(Constants.PARTITION_OPT).toString.toBoolean
    val localNegativeSample = if (partitionOpt) true else config.trainer.get(Constants.LOCAL_NEGATIVE_SAMPLE).toString.toBoolean
    val checkpointInterval = config.trainer.get(Constants.CHECKPOINT_INTERVAL).toString.toInt
    val useBalancePartition = false
    val outputEmbeddingByNodeType = config.trainer.get(Constants.OUTPUT_EMBEDDING_BY_NODE_TYPE).toString.toBoolean
    val outputEmbeddingByEdgeType = config.trainer.get(Constants.OUTPUT_EMBEDDING_BY_EDGE_TYPE).toString.toBoolean

    val conf = EntryUtils.start(mode, torchModelPath, "GATNE", actionType)
    //auto-adjust numPartitions and psNumPartition
    dataPartitionNum = PartitionUtils.getDataPartitionNum(dataPartitionNum, conf, dataPartitionNumFactor)
    psPartitionNum = PartitionUtils.getPsPartitionNum(psPartitionNum, conf, psPartitionNumFactor, featEmbeddingLoadPath)
    println(s"dataPartitionNum: $dataPartitionNum, psPartitionNum: $psPartitionNum")

    val gatne = new GATNE()
    gatne.setTorchModelPath(torchModelPath)
    gatne.setOptimizer(optimizer)
    gatne.setDataFormat(format)
    gatne.setBatchSize(batchSize)
    gatne.setStepSize(stepSize)
    gatne.setValidatePeriods(validatePeriods)
    gatne.setPSPartitionNum(psPartitionNum)
    gatne.setPartitionNum(dataPartitionNum)
    gatne.setUseBalancePartition(useBalancePartition)
    gatne.setNumEpoch(epoch)
    gatne.setStorageLevel(storageLevel)
    gatne.setSaveCheckpoint(saveCheckpoint)
    gatne.setCheckpointInterval(checkpointInterval)
    gatne.setDecay(decay)
    gatne.setEvaluations(evals)
    gatne.setContextDim(contextDim)
    gatne.setWindowSize(windowSize)
    gatne.setNegative(numNegSamples)
    gatne.setNegSampleByNodeType(negSampleByNodeType)
    gatne.setLocalNegativeSample(localNegativeSample)
    gatne.setSaveCheckpoint(saveCheckpoint)
    gatne.setSaveModelInterval(saveModelInterval)
    gatne.setInitMethod(iniFuncType)
    gatne.setEachNumSample(nodesSampleNum)
    gatne.setLogStep(logStep)
    gatne.setFieldMultiHot(fieldMultiHot)
    gatne.setFeatEmbedPath(featEmbeddingLoadPath)
    gatne.setContextEmbedPath(ContextEmbeddingInputPath)
    gatne.setOutputEmbeddingByNodeType(outputEmbeddingByNodeType)
    gatne.setOutputEmbeddingByEdgeType(outputEmbeddingByEdgeType)

    val featurePaths = parseNodePaths(config.graph.nodes)

    val data = GraphIO.loadString(walkInput)
    val edges = IOFunctions.loadEdge(edgeInput, isTyped = true, sep = sep)
    val testEdges = if (testEdgeInput.nonEmpty) Option(IOFunctions.loadEdgeWithLabel(testEdgeInput, isTyped = true, sep = sep)) else None
    val featuresMap = featurePaths.map { case (name, (path, sep)) =>
      (name.toInt, IOFunctions.loadFeature(path, sep = sep))
    }.toMap

    val nodeType = IOFunctions.loadNodeType(nodeTypeInput, sep=sep)

    val model = gatne.initialize(edges, featuresMap, nodeType, mean, std)
    edges.unpersist()
    gatne.showSummary(model)

    if (Constants.TRAIN.equals(actionType))
      gatne.fit(model, data, testEdges, outputModelPath, evaluateByEdgeType)

    if (embeddingOutputPath.nonEmpty) {
      gatne.saveEmbedding(model, embeddingOutputPath)
    }

    if (Constants.TRAIN.equals(actionType) && outputModelPath.nonEmpty) {
      gatne.save(model, outputModelPath + Constants.TORCH_MODEL_SAVE_SUB_DIR)
      gatne.saveFeatEmbeds_(model, outputModelPath)
      gatne.saveContext(model, outputModelPath)
    }

    EntryUtils.stop()
  }
}

