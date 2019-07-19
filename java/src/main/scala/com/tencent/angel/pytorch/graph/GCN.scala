package com.tencent.angel.pytorch.graph

import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.params._
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.ml.graph.params._
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.storage.StorageLevel

class GCN(override val uid: String) extends Transformer
  with HasSrcNodeIdCol with HasDstNodeIdCol with HasPartitionNum with HasPSPartitionNum
  with HasInputFeaturePath with HasTorchModelPath with HasBatchSize with HasLabelPath with HasFeatureDim
  with HasOptimizer with HasUseBalancePartition with HasStorageLevel with HasNumEpoch with HasNumSamples
  with HasDataFormat {

  def this() = this(Identifiable.randomUID("GCN"))

  override def transform(dataset: Dataset[_]): DataFrame = {

    val (model, graph, optim) = initialize(dataset)

    val numNodes = model.numNodes()
    println(s"numNodes=$numNodes")

    for (curEpoch <- 1 to $(numEpoch)) {
      val (lossSum, totalTrain) = graph.map(_.trainEpoch(curEpoch, $(batchSize), model,
        $(featureDim), optim, $(numSamples))).reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
      val (right, totalPred) = graph.map(_.predictEpoch(curEpoch, $(batchSize), model,
        $(featureDim), $(numSamples))).reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
      println(s"curEpoch=$curEpoch loss=${lossSum / totalTrain} precision=${right.toDouble / totalPred}")
    }

    null

  }


  def initialize(dataset: Dataset[_]): (GraphPSModel, RDD[GraphMiniBatchGCNPartition], AsyncOptim) = {
    val start = System.currentTimeMillis()

    // read edges
    val edges = dataset.select($(srcNodeIdCol), $(dstNodeIdCol)).rdd
      .map(row => (row.getLong(0), row.getLong(1)))
      .filter(f => f._1 != f._2)

    edges.persist(StorageLevel.DISK_ONLY)

    val (minId, maxId, numEdges) = edges.mapPartitions(summarizeApplyOp).reduce(summarizeReduceOp)
    val index = edges.flatMap(f => Iterator(f._1, f._2))
    println(s"minId=$minId maxId=$maxId numEdges=$numEdges")

    // create weights, graph on servers
    TorchModel.setPath($(torchModelPath))
    val weightsSize = TorchModel.get().getParametersTotalSize
    println(s"weight total size=${weightsSize}")
    val optim = getOptimizer

    val model = GraphPSModel.apply(minId, maxId + 1, weightsSize, optim,
      index, $(psPartitionNum), $(useBalancePartition))

    // read labels
    readLabels($(labelPath), model)

    // read features
    readFeatures($(inputFeaturePath), $(featureDim), model)

    // build adj graph partitions
    val adjGraph = edges.map(sd => (sd._1, sd._2)).groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(GraphAdjPartition.apply(index, it)))

    adjGraph.persist($(storageLevel))
    adjGraph.foreachPartition(_ => Unit)
    adjGraph.map(_.init(model)).reduce(_ + _)

    // build GCN graph partitions
    val gcnGraph = adjGraph.map(_.toMiniBatchGCNPartition(model, $(torchModelPath)))
    gcnGraph.persist($(storageLevel))
    gcnGraph.count()
    adjGraph.unpersist(true)

    // initialize weights with torch values
    model.setWeights(TorchModel.get().getParameters)

    val end = System.currentTimeMillis()
    println(s"initialize cost ${(end - start) / 1000}s")

    (model, gcnGraph, optim)
  }

  def readLabels(path: String, model: GraphPSModel): Unit = {
    val labels = SparkContext.getOrCreate().textFile(path)
      .map(f => f.split(" "))
      .map(f => (f(0).toLong, f(1).toFloat))

    labels.mapPartitionsWithIndex((index, it) =>
      Iterator.single(NodeLabelPartition.apply(index, it, model.dim)))
      .map(_.init(model)).count()
  }

  def readFeatures(path: String, dim: Int, model: GraphPSModel): Unit = {
    val features = SparkContext.getOrCreate().textFile(path)
      .map(f => SampleParser.parseNodeFeature(f, dim, $(dataFormat)))
      .map(f => (f._1.toLong, f._2))
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(NodeFeaturePartition.apply(index, it)))

    features.map(_.init(model)).count()
  }

  def summarizeApplyOp(iterator: Iterator[(Long, Long)]): Iterator[(Long, Long, Long)] = {
    var minId = Long.MaxValue
    var maxId = Long.MinValue
    var numEdges = 0
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (src, dst) = (entry._1, entry._2)
      minId = math.min(minId, src)
      minId = math.min(minId, dst)
      maxId = math.max(maxId, src)
      maxId = math.max(maxId, dst)
      numEdges += 1
    }

    Iterator.single((minId, maxId, numEdges))
  }

  def summarizeReduceOp(t1: (Long, Long, Long),
                        t2: (Long, Long, Long)): (Long, Long, Long) =
    (math.min(t1._1, t2._1), math.max(t1._2, t2._2), t1._3 + t2._3)


  // currently just return a fake scheme
  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(
      StructField(s"temp", LongType, nullable = false)
    ))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
