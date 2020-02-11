package com.tencent.angel.pytorch.graph.encode

import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.pytorch.io.DataLoaderUtils
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.graph.params.{HasDstNodeIdCol, HasIsWeighted, HasPartitionNum, HasSrcNodeIdCol, HasWeightCol}
import it.unimi.dsi.fastutil.longs.LongOpenHashSet
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{FloatType, IntegerType, LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

class Encode(override val uid: String) extends Transformer
  with HasSrcNodeIdCol with HasDstNodeIdCol with HasIsWeighted with HasWeightCol
  with HasPartitionNum {

  def this() = this(Identifiable.randomUID("Encode"))

  override
  def transform(dataset: Dataset[_]): DataFrame = {
    val edges_ = if ($(isWeighted)) {
      dataset.select(${srcNodeIdCol}, ${dstNodeIdCol}, ${weightCol}).rdd
        .map(row => (row.getLong(0), row.getLong(1), row.getFloat(2)))
        .filter(f => f._1 != f._2)
        .filter(f => f._3 != 0)
    } else {
      dataset.select(${srcNodeIdCol}, ${dstNodeIdCol}).rdd
        .map(row => (row.getLong(0), row.getLong(1), 1.0f))
        .filter(f => f._1 != f._2)
    }

    val edges = edges_.repartition($(partitionNum))

    val (minId, maxId, numEdges) = edges.map(f => (f._1, f._2))
      .mapPartitions(DataLoaderUtils.summarizeApplyOp)
      .reduce(DataLoaderUtils.summarizeReduceOp)

    println(s"minId=$minId maxId=$maxId numEdges=$numEdges")

    val nodes = edges.flatMap(f => Iterator(f._1, f._2)).distinct()
    val nodesWithIndex = nodes.zipWithIndex()

    nodesWithIndex.persist(StorageLevel.DISK_ONLY)
    val count = nodesWithIndex.count()
    println(s"count=$count")

    PSContext.getOrCreate(SparkContext.getOrCreate())
    val model = EncodePSModel.fromMinMax(minId, maxId + 1)
    initIndex(nodesWithIndex, model)
    val result = encodeEdges(edges, model)
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3)))
    val outputSchema = transformSchema(dataset.schema)
    dataset.sparkSession.createDataFrame(result, outputSchema)

  }

  def initIndex(nodes: RDD[(Long, Long)], model: EncodePSModel): Unit = {

    def init(index: Int, it: Iterator[(Long, Long)]): Iterator[Int] = {
      val update = VFactory.sparseLongKeyIntVector(model.dim)
      it.foreach(f => update.set(f._1, f._2.toInt))
      model.initIndex(update)
      Iterator(0)
    }

    nodes.mapPartitionsWithIndex(init).reduce(_ + _)
  }

  def encodeEdges(edges: RDD[(Long, Long, Float)], model: EncodePSModel): RDD[(Int, Int, Float)] = {
    def encode(index: Int, it: Iterator[(Long, Long, Float)]): Iterator[(Int, Int, Float)] = {
      val keys = new LongOpenHashSet()
      val arrays = it.toArray
      arrays.foreach { f =>
        keys.add(f._1)
        keys.add(f._2)
      }

      val map = model.readIndex(keys.toLongArray)
      arrays.map { f =>
        (map.get(f._1), map.get(f._2), f._3)
      }.iterator
    }

    edges.mapPartitionsWithIndex(encode)
  }

  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(
      StructField(s"${$(srcNodeIdCol)}", IntegerType, nullable = false),
      StructField(s"${$(dstNodeIdCol)}", IntegerType, nullable = false),
      StructField(s"weight", FloatType, nullable = false)
    ))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
