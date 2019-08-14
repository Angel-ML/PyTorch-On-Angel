package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.graph.utils.DataLoaderUtils
import com.tencent.angel.pytorch.params._
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.graph.params._
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

class GNN(val uid: String) extends Serializable
  with HasTorchModelPath with HasBatchSize with HasFeatureDim
  with HasOptimizer with HasNumEpoch with HasNumSamples with HasNumBatchInit
  with HasPartitionNum with HasPSPartitionNum with HasUseBalancePartition
  with HasDataFormat with HasStorageLevel {

  def this() = this(Identifiable.randomUID("GNN"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  def fit(model: GNNPSModel, graph: Dataset[_]): Unit = ???

  def initFeatures(model: GNNPSModel, features: Dataset[Row], minId: Long, maxId: Long): Unit = {
    features.rdd.filter(row => row.length > 0)
      .filter(row => row.get(0) != null)
      .map(row => (row.getLong(0), row.getString(1)))
      .filter(f => f._1 >= minId && f._1 <= maxId)
      .map(f => (f._1, SampleParser.parseFeature(f._2, $(featureDim), $(dataFormat))))
      .mapPartitionsWithIndex((index, it) =>
        Iterator(NodeFeaturePartition.apply(index, it)))
      .map(_.init(model, $(numBatchInit))).count()
  }

  def initLabels(model: GNNPSModel, labels: Dataset[Row], minId: Long, maxId: Long): Unit = {
    labels.rdd.map(row => (row.getLong(0), row.getFloat(1)))
      .filter(f => f._1 >= minId && f._1 <= maxId)
      .mapPartitionsWithIndex((index, it) =>
        Iterator(NodeLabelPartition.apply(index, it, model.dim)))
      .map(_.init(model)).count()
  }

  def makeGraph(edges: RDD[(Long, Long)], model: GNNPSModel): Dataset[_] = ???

  def initialize(edgeDF: DataFrame, featureDF: DataFrame): (GNNPSModel, Dataset[_]) =
    initialize(edgeDF, featureDF, None)

  def initialize(edgeDF: DataFrame,
                 featureDF: DataFrame,
                 labelDF: Option[DataFrame]): (GNNPSModel, Dataset[_]) = {
    val start = System.currentTimeMillis()

    // read edges
    val edges = edgeDF.select("src", "dst").rdd
      .map(row => (row.getLong(0), row.getLong(1)))
      .filter(f => f._1 != f._2)

    edges.persist(StorageLevel.DISK_ONLY)

    val (minId, maxId, numEdges) = edges.mapPartitions(DataLoaderUtils.summarizeApplyOp)
      .reduce(DataLoaderUtils.summarizeReduceOp)
    val index = edges.flatMap(f => Iterator(f._1, f._2))
    println(s"minId=$minId maxId=$maxId numEdges=$numEdges")

    // create weights, graph on servers
    TorchModel.setPath($(torchModelPath))
    val torch = TorchModel.get()
    val weightsSize = torch.getParametersTotalSize
    println(s"weight total size=${weightsSize}")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    val model = GNNPSModel.apply(minId, maxId + 1, weightsSize, getOptimizer,
      index, $(psPartitionNum), $(useBalancePartition))

    // initialize weights with torch values
    model.setWeights(torch.getParameters)
    TorchModel.addModel(torch)

    labelDF.foreach(f => initLabels(model, f, minId, maxId))
    initFeatures(model, featureDF, minId, maxId)

    val graph = makeGraph(edges, model)

    val end = System.currentTimeMillis()
    println(s"initialize cost ${(end - start) / 1000}s")

    (model, graph)
  }


  def genEmbedding(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[GNNPartition]
      .genEmbedding($(batchSize) * 10, model, $(featureDim), $(numSamples)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  def genLabels(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[GNNPartition]
      .genLabels($(batchSize) * 10, model, $(featureDim), $(numSamples)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", LongType, nullable = false),
      StructField("softmax", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }


  def showSummary(model: GNNPSModel, graph: Dataset[_]): Unit = {
    println(s"numNodesHasLabels=${model.nnzLabels()}")
    println(s"nnzFeatures=${model.nnzFeatures()}")
    println(s"nnzNode=${model.nnzNodes()}")
    println(s"nnzEdge=${model.nnzEdge()}")
    println(s"nnzNeighbor=${model.nnzNeighbors()}")
    val numNodesWithOutEdges = graph.rdd.map(_.asInstanceOf[GNNPartition].numNodes).reduce(_ + _)
    val numEdges = graph.rdd.map(_.asInstanceOf[GNNPartition].numEdges).reduce(_ + _)
    println(s"numNodesWithOutEdges=$numNodesWithOutEdges")
    println(s"numEdges=$numEdges")
  }

}
