package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.pytorch.eval.{Evaluation, EvaluationM}
import com.tencent.angel.pytorch.io.DataLoaderUtils
import com.tencent.angel.pytorch.params._
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.context.PSContext
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.util.Random


class GAMLP extends GCN {

  def initialize(featureDF: DataFrame, labelDF: Option[DataFrame], testLabelDF: Option[DataFrame]): (GNNPSModel, Dataset[_]) = {

    val start = System.currentTimeMillis()
    val nodes = featureDF.select("node").rdd.map(_.getLong(0)).persist($(storageLevel))
    val (minId, maxId, numNodes) = nodes.mapPartitions(DataLoaderUtils.summarizeNodesApplyOp)
      .reduce(DataLoaderUtils.summarizeReduceOp)
    println(s"minId=$minId maxId=$maxId numSamples=$numNodes")

    // create weights, graph on servers
    TorchModel.setPath($(torchModelPath))
    println("get torch model")
    val torch = TorchModel.get()
    println("get torch model done, get parameters")
    val weightsSize = torch.getParametersTotalSize
    println(s"weight total size=$weightsSize")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    // init ps model
    val model = if ($(fieldNum) > 0) {
      SparseGNNPSModel(minId, maxId + 1, weightsSize, getOptimizer, nodes,
        $(psPartitionNum), $(useBalancePartition), $(featEmbedDim), $(featureDim))
    } else {
      GNNPSModel.apply(minId, maxId + 1, weightsSize, getOptimizer,
        nodes, $(psPartitionNum), $(useBalancePartition))
    }
    nodes.unpersist()

    // initialize weights with torch values
    model.setWeights(torch.getParameters)

    // init label to ps
    if ($(numLabels) == 1) {
      // init labels to labels and testLabels PSVectors
      labelDF.foreach(f => initLabels(model, f, minId, maxId))
      testLabelDF.foreach(f => initTestLabels(model, f, minId, maxId))
    } else {
      // init label arrays to userGraph PSMatrix
      labelDF.foreach(f => initMultiLabels(model, f, minId, maxId))
      testLabelDF.foreach(f => initMultiTestLabels(model, f, minId, maxId))
    }

    val graph = makeGraph(featureDF, model, labelDF, testLabelDF, minId, maxId)

    // correct featureDim for sparse input after initFeatures
    if ($(fieldNum) > 0) {
      setFeatureDim($(featEmbedDim))
    } else {
      setFeatureDim($(featureDim))
    }
    TorchModel.put(torch)

    val end = System.currentTimeMillis()
    println(s"initialize cost ${(end - start) / 1000}s")
    val startTs = System.currentTimeMillis()
    if ($(saveCheckpoint))
      model.checkpointMatrices(0)
    println(s"Write checkpoint use time=${System.currentTimeMillis() - startTs}ms")
    (model, graph)
  }

  override
  def makeGraph(featureDF: DataFrame, model: GNNPSModel, labelDF: Option[DataFrame],
                testLabelDF: Option[DataFrame], minId: Long, maxId: Long): Dataset[_] = {
    // build feature partitions
    val dataParts =
      featureDF.select("node", "feature")
        .filter(!_.anyNull).rdd
        .map(row => (row.getLong(0), row.getString(1)))
        .repartition($(partitionNum))
        .mapPartitionsWithIndex((index, it) => Iterator.single(
          FeaturePartition(index, it, $(featureDim), $(dataFormat), $(testRatio), $(numLabels), $(torchModelPath))))

    dataParts.persist($(storageLevel))
    dataParts.foreachPartition(_ -> Unit)
    val gcnGraph = dataParts.map(_.toSemiGAMLPPartition(model, labelDF.nonEmpty))
    gcnGraph.persist($(storageLevel))
    gcnGraph.foreachPartition(_ -> Unit)
    dataParts.unpersist()

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[GAMLPPartition]
    SparkSession.builder().getOrCreate().createDataset(gcnGraph)
  }

  override
  def genLabels(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[GAMLPPartition]
        .genLabels($(batchSize), model,  $(featureDim), $(numSamples), $(partitionNum), $(numLabels), true, $(fieldNum), false))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", StringType, nullable = false),
      StructField("softmax", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  override
  def evaluate(model: GNNPSModel, graph: Dataset[_], isTest: Boolean = true): Map[String, String] = {
    import com.tencent.angel.pytorch.eval.Evaluation._
    val scores = graph.rdd.flatMap(_.asInstanceOf[GAMLPPartition]
        .predictEpoch(0, $(batchSize) * $(batchSizeMultiple), model, $(featureDim), $(numSamples), isTest, $(fieldNum), $(fieldMultiHot)))
      .flatMap(f => f._1.zip(f._2))
      .persist(StorageLevel.MEMORY_ONLY)
    if ($(numLabels) > 1) EvaluationM.eval(getEvaluations, scores, $(numLabels))
    else Evaluation.eval(getEvaluations, scores).map(x => (x._1, x._2.toString))
  }
}