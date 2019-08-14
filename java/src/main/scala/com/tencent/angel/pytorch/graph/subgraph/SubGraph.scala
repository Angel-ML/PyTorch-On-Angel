package com.tencent.angel.pytorch.graph.subgraph

import com.tencent.angel.pytorch.graph.utils.DataLoaderUtils
import com.tencent.angel.pytorch.params.{HasInputFeaturePath, HasLabelPath, HasOutputFeaturePath}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.graph.params._
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class SubGraph(override val uid: String) extends Transformer
  with HasSrcNodeIdCol with HasDstNodeIdCol with HasPartitionNum with HasPSPartitionNum
  with HasLabelPath with HasInputFeaturePath with HasUseBalancePartition with HasStorageLevel
  with HasOutputFeaturePath {

  val numBatchInit = new IntParam(this, "numBatch", "numBatch")
  setDefault(numBatchInit, 4)

  def setNumBatchInit(num: Int): this.type = set(numBatchInit, num)

  def this() = this(Identifiable.randomUID("SubGraph"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val edges = dataset.select($(srcNodeIdCol), $(dstNodeIdCol)).rdd
      .map(row => (row.getLong(0), row.getLong(1)))
      .filter(f => f._1 != f._2)

    val (minId, maxId, numEdges) = edges.mapPartitions(DataLoaderUtils.summarizeApplyOp)
      .reduce(DataLoaderUtils.summarizeReduceOp)
    val index = edges.flatMap(f => Iterator(f._1, f._2))
    println(s"minId=$minId maxId=$maxId numEdges=$numEdges")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    val model = SubGraphPSModel.apply(minId, maxId + 1, index,
      $(psPartitionNum), $(useBalancePartition))

    // init nodes with labels
    readLabels($(labelPath), model)

    // build graph
    val graph = edges.groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator(SubGraphPSPartition.apply(index, it)))

    graph.persist($(storageLevel))
    graph.count()

    graph.map(_.init(model, $(numBatchInit))).count()
    graph.map(_.sample(model)).count()

    sampleFeatures($(inputFeaturePath), $(outputFeaturePath), model)

    val twoOrderEdges = graph.flatMap(_.twoOrderEdges(model))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val outputSchema = transformSchema(dataset.schema)
    dataset.sparkSession.createDataFrame(twoOrderEdges, outputSchema)
  }

  def readLabels(path: String, model: SubGraphPSModel): Unit = {

    def initLabels(iterator: Iterator[Long]): Iterator[Int] = {
      val keys = iterator.toArray
      model.initNodesWithLabels(keys)
      Iterator(0)
    }

    SparkContext.getOrCreate().textFile(path)
      .map(f => f.split(" ")(0).toLong)
      .mapPartitions(initLabels)
      .count()

  }

  def sampleFeatures(inputPath: String, outputPath: String, model: SubGraphPSModel): Unit = {
    val features = SparkContext.getOrCreate().textFile(inputPath)
      .mapPartitionsWithIndex((index, it) => Iterator(FeaturePartition.apply(index, it)))

    features.flatMap(_.sample(model))
      .map(f => s"${f._1} ${f._2}")
      .saveAsTextFile(outputPath)
  }

  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(
      StructField(s"${$(srcNodeIdCol)}", LongType, nullable = false),
      StructField(s"${$(dstNodeIdCol)}", LongType, nullable = false)
    ))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)


}
