package com.tencent.angel.pytorch.feature.normalize

import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.storage.{IntFloatDenseVectorStorage, IntFloatSortedVectorStorage}
import com.tencent.angel.ml.math2.ufuncs.Ufuncs
import com.tencent.angel.ml.math2.vector.IntFloatVector
import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.params.{HasDataFormat, HasFeatureDim}
import com.tencent.angel.spark.context.PSContext
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

class MeanVarNormalize(override val uid: String) extends Transformer
  with HasFeatureDim with HasDataFormat {

  def this() = this(Identifiable.randomUID("Normalize"))

  override def transform(dataset: Dataset[_]): DataFrame = {

    val count = dataset.count()
    println(s"count=$count")

    val features = dataset.select("node", "feature")
      .rdd.filter(row => row.length > 0)
      .filter(row => row.get(0) != null)
      .map(row => (row.getLong(0), row.getString(1)))
      .map(f => (f._1, SampleParser.parseFeature(f._2, $(featureDim), $(dataFormat))))

    features.persist(StorageLevel.DISK_ONLY)
    val numFeatures = features.count()
    println(s"numFeatures=$numFeatures")

    PSContext.getOrCreate(SparkContext.getOrCreate())
    val model = MeanVarNormalizePSModel.apply(0, $(featureDim))

    def calculateSum(iterator: Iterator[(_, IntFloatVector)]): Iterator[Int] = {
      val sum = VFactory.denseFloatVector($(featureDim))
      while (iterator.hasNext) {
        val f = iterator.next()._2
        sum.iadd(f)
      }

      model.addSum(sum)
      Iterator(0)
    }

    def calculateVariance(iterator: Iterator[(_, IntFloatVector)]): Iterator[Int] = {
      val mean = model.readMean()
      val sumVar = VFactory.denseFloatVector($(featureDim))

      while (iterator.hasNext) {
        val f = iterator.next()._2
        f.isub(mean)
        Ufuncs.ipow(f, 2)
        sumVar.iadd(f)
      }

      model.addVar(sumVar)
      Iterator(0)
    }

    def featureToString(f: IntFloatVector): String = {
      f.getStorage match {
        case sorted: IntFloatSortedVectorStorage =>
          val indices = sorted.getIndices
          val values = sorted.getValues
          for (i <- values.indices)
            if (values(i).isNaN)
              values(i) = 0.0f
          val ff = indices.zip(values).map(kv => s"${kv._1}:${kv._2}")
          ff.mkString(" ")
        case dense: IntFloatDenseVectorStorage =>
          val values = dense.getValues
          for (i <- values.indices)
            if (values(i).isNaN)
              values(i) = 0.0f
          values.mkString(" ")
      }
    }


    def normalize(iterator: Iterator[(Long, IntFloatVector)]): Iterator[(Long, IntFloatVector)] = {
      val mean = model.readMean()
      val variance = model.readVariance()
      iterator.map { case (key, f) =>
        f.isub(mean).idiv(variance)
        (key.toLong, f)
      }
    }

    features.mapPartitions(calculateSum).reduce(_ + _)
    model.calculateMean(count)
    features.mapPartitions(calculateVariance).reduce(_ + _)
    model.calculateVar(count)

    val output = features.mapPartitions(normalize)
      .map(f => (f._1, featureToString(f._2)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    dataset.sparkSession.createDataFrame(output,
      transformSchema(dataset.schema))

  }

  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("feature", StringType, nullable = false)
    ))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
