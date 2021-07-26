package com.tencent.angel.pytorch.graph.gcn

import java.io.File

import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.io.DataLoaderUtils
import com.tencent.angel.pytorch.params._
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.context.PSContext
import org.apache.hadoop.fs.permission.FsPermission
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.{SparkContext, SparkEnv}

import scala.language.existentials

class BiSAGE extends GNN with HasUserFeatureDim with HasItemFeatureDim with HasPSModelCheckpoint
  with HasUserFieldNum with HasItemFieldNum with HasUserFeatEmbedDim with HasItemFeatEmbedDim
  with HasFieldMultiHot with HasTestRatio with HasUserNumSamples with HasItemNumSamples {


  def initialize(edgeDF: DataFrame, userFeatureDF: DataFrame, itemFeatureDF: DataFrame): (BiSAGEPSModel, Dataset[_], Dataset[_]) =
    initialize(edgeDF, userFeatureDF, itemFeatureDF, None, None)

  def initialize(edgeDF: DataFrame,
                 userFeatureDF: DataFrame,
                 itemFeatureDF: DataFrame,
                 labelDF: Option[DataFrame],
                 testLabelDF: Option[DataFrame]): (BiSAGEPSModel, Dataset[_], Dataset[_]) = {
    val start = System.currentTimeMillis()

    // read edges
    edgeDF.persist($(storageLevel))

    val (userMinId, userMaxId, numEdges1) = getMinMaxId(edgeDF, "src")
    println(s"userMinId=$userMinId userMaxId=$userMaxId numEdges=$numEdges1")
    val (itemMinId, itemMaxId, numEdges2) = getMinMaxId(edgeDF, "dst")
    println(s"itemMinId=$itemMinId itemMaxId=$itemMaxId numEdges=$numEdges2")

    // create weights, graph on servers
    TorchModel.setPath($(torchModelPath))
    val torch = TorchModel.get()
    val weightsSize = torch.getParametersTotalSize
    println(s"weight total size=${weightsSize}")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    val model =
      if ($(userFieldNum) > 0)
        SparseBiSAGEPSModel.apply(userMinId, userMaxId + 1, itemMinId, itemMaxId + 1, weightsSize, getOptimizer,
          getPartitionIndex(edgeDF), $(psPartitionNum), $(useBalancePartition), $(userFeatEmbedDim),
          $(userFeatureDim), $(itemFeatEmbedDim), $(itemFeatureDim))
      else
        BiSAGEPSModel.apply(userMinId, userMaxId + 1, itemMinId, itemMaxId + 1, weightsSize, getOptimizer,
          getPartitionIndex(edgeDF), $(psPartitionNum), $(useBalancePartition))

    // initialize weights with torch values
    model.setWeights(torch.getParameters)
    TorchModel.put(torch)

    // initialize embeddings
    if ($(userFieldNum) > 0) {
      if ($(featEmbedPath).length > 0 ) {
        println(s"load sparse feature embedding from ${featEmbedPath}.")
        model.asInstanceOf[SparseBiSAGEPSModel].loadFeatEmbed($(featEmbedPath))
      }
      else {
        println(s"init sparse feature embedding.")
        model.asInstanceOf[SparseBiSAGEPSModel].initUserEmbeddings()
        if ($(itemFeatureDim) > 0)
          model.asInstanceOf[SparseBiSAGEPSModel].initItemEmbeddings()
      }
    }

    labelDF.foreach(f => initLabels(model, f, userMinId, userMaxId))
    testLabelDF.foreach(f => initTestLabels(model, f, userMinId, userMaxId))

    val (userGraph, itemGraph) = makeGraph(edgeDF, model)

    initFeatures(model, userFeatureDF, itemFeatureDF, userMinId, userMaxId, itemMinId, itemMaxId)

    // correct featureDim for sparse input after initFeatures
    if ($(userFieldNum) > 0) {
      setUserFeatureDim($(userFeatEmbedDim))
      if ($(itemFeatureDim) > 0) setItemFeatureDim($(itemFeatEmbedDim))
    }

    val end = System.currentTimeMillis()
    println(s"initialize cost ${(end - start) / 1000}s")
    val startTs = System.currentTimeMillis()
    if ($(saveCheckpoint))
      model.checkpointMatrices(0)
    println(s"Write checkpoint use time=${System.currentTimeMillis() - startTs}ms")
    (model, userGraph, itemGraph)
  }

  def initFeatures(model: BiSAGEPSModel,
                   userFeatures: Dataset[Row],
                   itemFeatures: Dataset[Row],
                   userMinId: Long, userMaxId: Long,
                   itemMinId: Long, itemMaxId: Long): Unit = {
    if ($(userFeatureDim) > 0)
      userFeatures.select("node", "feature").rdd.filter(row => row.length > 0)
        .filter(row => row.get(0) != null)
        .map(row => (row.getLong(0), row.getString(1)))
        .filter(f => f._1 >= userMinId && f._1 <= userMaxId)
        .map(f => (f._1, SampleParser.parseFeature(f._2, $(userFeatureDim), $(dataFormat))))
        .repartition($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator.single(NodeFeaturePartition.apply(index, it)))
        .map(_.init(model, 0, $(numBatchInit))).count()  // 0 for init features on userGraph

    if ($(itemFeatureDim) > 0)
      itemFeatures.select("node", "feature").rdd.filter(row => row.length > 0)
        .filter(row => row.get(0) != null)
        .map(row => (row.getLong(0), row.getString(1)))
        .filter(f => f._1 >= itemMinId && f._1 <= itemMaxId)
        .map(f => (f._1, SampleParser.parseFeature(f._2, $(itemFeatureDim), $(dataFormat))))
        .repartition($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator.single(NodeFeaturePartition.apply(index, it)))
        .map(_.init(model, 1, $(numBatchInit))).count()     // 1 for init features on itemGraph
  }

  def getMinMaxId(edges: DataFrame, nodeTypes: String): (Long, Long, Long) =
  //nodeTypes: src or dst
    edges.select(nodeTypes).rdd
      .map(row => (row.getLong(0), row.getLong(0)))   // count only one type
      .mapPartitions(DataLoaderUtils.summarizeApplyOp)
      .reduce(DataLoaderUtils.summarizeReduceOp)

  def showSummary(model: BiSAGEPSModel, userGraph: Dataset[_], itemGraph: Dataset[_]): Unit = {
    println(s"numNodesHasLabels=${model.nnzLabels()}")
    println(s"numTestLabels=${model.nnzTestLabels()}")
    println(s"nnzUserFeatures=${model.nnzFeatures(0)}")
    println(s"nnzItemFeatures=${model.nnzFeatures(1)}")
    println(s"nnzUserNode=${model.nnzNodes(0)}")
    println(s"nnzItemNode=${model.nnzNodes(1)}")
    println(s"nnzEdge=${model.nnzEdge(0)}")
    println(s"nnzUserNeighbor=${model.nnzNeighbors(0)}")
    println(s"nnzItemNeighbor=${model.nnzNeighbors(1)}")
    val numUserNodesHasOutEdges = userGraph.rdd.map(_.asInstanceOf[GNNPartition].numNodes).reduce(_ + _)
    val numItemNodesHasOutEdges = itemGraph.rdd.map(_.asInstanceOf[GNNPartition].numNodes).reduce(_ + _)
    val numEdges = userGraph.rdd.map(_.asInstanceOf[GNNPartition].numEdges).reduce(_ + _)
    println(s"numUserNodesWithOutEdges=$numUserNodesHasOutEdges")
    println(s"numItemNodesWithOutEdges=$numItemNodesHasOutEdges")
    println(s"numEdges=$numEdges")
  }

  def genEmbedding(model: BiSAGEPSModel, graph: Dataset[_], graphType: Int): DataFrame = {
    val alpha = if ($(batchSize) >= 5000) 1 else $(batchSizeMultiple)
    val ret = graph.rdd.flatMap(_.asInstanceOf[BiSAGEPartition]
      .genEmbedding($(batchSize) * alpha, model, $(userFeatureDim), $(itemFeatureDim), $(userNumSamples), $(itemNumSamples), graph.rdd.getNumPartitions, graphType, $(userFieldNum), $(itemFieldNum), $(fieldMultiHot)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  def makeGraph(edges: DataFrame, model: BiSAGEPSModel): (Dataset[_], Dataset[_]) = {
    val adj_u = edges.select("src", "dst").rdd
      .map(row => (row.getLong(0), (row.getLong(1), 0, 0)))
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator(GraphAdjBiPartition.apply(index, it, 0))) // 0 for init neighbors on userGraph

    val adj_i = edges.select("src", "dst").rdd
      .map(row => (row.getLong(1), (row.getLong(0), 0, 0)))
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator(GraphAdjBiPartition.apply(index, it, 1))) // 1 for init neighbors on itemGraph

    adj_u.persist($(storageLevel))
    adj_i.persist($(storageLevel))
    adj_u.foreachPartition(_ => Unit)
    adj_i.foreachPartition(_ => Unit)
    // init neighbors on PS
    adj_u.map(_.init(model, $(numBatchInit))).reduce(_ + _)
    adj_i.map(_.init(model, $(numBatchInit))).reduce(_ + _)

    val userGraph = adj_u.map(_.toBiSAGEPartition(model, $(torchModelPath), $(useSecondOrder), $(dataFormat)))

    val itemGraph = adj_i.map(_.toBiSAGEPartition(model, $(torchModelPath), $(useSecondOrder), $(dataFormat)))

    userGraph.persist($(storageLevel))
    userGraph.foreachPartition(_ => Unit)
    itemGraph.persist($(storageLevel))
    itemGraph.foreachPartition(_ => Unit)
    adj_u.unpersist(true)
    adj_i.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[BiSAGEPartition]
    (SparkSession.builder().getOrCreate().createDataset(userGraph),
      SparkSession.builder().getOrCreate().createDataset(itemGraph))

  }

  def fit(model: BiSAGEPSModel, userGraph: Dataset[_], itemGraph: Dataset[_]): Unit = {
    fit(model, userGraph, itemGraph, checkPointPath = null)
  }

  def fit(model: BiSAGEPSModel, userGraph: Dataset[_], itemGraph: Dataset[_], checkPointPath: String): Unit = {
    val optim = getOptimizer

    for (curEpoch <- 1 to $(numEpoch)) {
      var start = System.currentTimeMillis()
      val (lossSum1, totalTrain1, numSteps1) = userGraph.rdd.map(_.asInstanceOf[BiSAGEPartition]
        .trainEpoch(curEpoch, $(batchSize), model, $(userFeatureDim), $(itemFeatureDim), optim, $(userNumSamples), $(itemNumSamples), 0, $(userFieldNum), $(itemFieldNum), $(fieldMultiHot), $(testRatio)))
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2, math.max(f1._3, f2._3)))

      val (lossSum2, totalTrain2, numSteps2) = itemGraph.rdd.map(_.asInstanceOf[BiSAGEPartition]
        .trainEpoch(curEpoch, $(batchSize), model, $(userFeatureDim), $(itemFeatureDim), optim, $(userNumSamples), $(itemNumSamples), 1, $(userFieldNum), $(itemFieldNum), $(fieldMultiHot), $(testRatio)))
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2, math.max(f1._3, f2._3)))

      optim.step(math.max(numSteps1, numSteps2))
      println(s"curEpoch=$curEpoch lr=${optim.getCurrentEta.toFloat} train loss=${(lossSum1 + lossSum2) / (totalTrain1 + totalTrain2)}, cost: ${(System.currentTimeMillis() - start) / 1000.0f} s")

      //optim.step(numSteps1)
      //println(s"curEpoch=$curEpoch train loss=${(lossSum1) / (totalTrain1)}")

      if (checkPointPath != null && checkPointPath.length > 0 && curEpoch % $(periods) == 0)
        save(model, checkPointPath, curEpoch)
    }

  }

  def save(model: BiSAGEPSModel, path: String): Unit = {
    // save pt to local file
    save(model, path, -1)
  }

  def save(model: BiSAGEPSModel, path: String, epoch: Int): Unit = {

    def getDestinationPath: Path = {
      if (epoch < 0)
        new Path(path + "/" + $(torchModelPath))
      else
        new Path(path + "/" + s"${$(torchModelPath)}.${epoch}")
    }

    // save pt to local file with epoch_id
    val weights = model.readWeights()
    val torch = TorchModel.get()
    torch.gcnSave(s"gcn-train-temp-${epoch}.pt", weights)

    val hdfsPath = new Path(path)
    val conf = SparkHadoopUtil.get.newConfiguration(SparkEnv.get.conf)
    val fs = hdfsPath.getFileSystem(conf)
    val outputPath = new Path(fs.makeQualified(hdfsPath).toUri.getPath)
    if (!fs.exists(outputPath)) {
      val permission = new FsPermission(FsPermission.createImmutable(0x1ff.toShort))
      FileSystem.mkdirs(fs, outputPath, permission)
    }

    val file = new File(s"gcn-train-temp-${epoch}.pt")
    if (file.exists()) {
      val srcPath = new Path(file.getPath)
      val dstPath = getDestinationPath
      fs.copyFromLocalFile(srcPath, dstPath)
      println(s"save model to ${dstPath.toString}")
    }
    TorchModel.put(torch)
  }

  /**
    * Write checkpoint or result if need
    *
    * @param epoch
    */
  def checkpointIfNeed(model: BiSAGEPSModel, epoch: Int): Unit = {
    var startTs = 0L
    if ($(checkpointInterval) > 0 && epoch % $(checkpointInterval) == 0 && epoch < $(numEpoch)) {
      println(s"Epoch=${epoch}, checkpoint the model")
      startTs = System.currentTimeMillis()
      model.checkpointMatrices(epoch+1)
      println(s"checkpoint use time=${System.currentTimeMillis() - startTs}ms")
    }
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}