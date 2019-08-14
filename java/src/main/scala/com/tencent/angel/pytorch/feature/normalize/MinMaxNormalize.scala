package com.tencent.angel.pytorch.feature.normalize

import java.lang.{Long => JLong}

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
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class MinMaxNormalize(override val uid: String) extends Transformer
  with HasFeatureDim with HasDataFormat {

  def this() = this(Identifiable.randomUID("Normalize"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    PSContext.getOrCreate(SparkContext.getOrCreate())
    val model = MinMaxNormalizePSModel.apply(0, $(featureDim))

    val features = dataset.select("feature").rdd.map(row => row.getString(0))
      .map(line => SampleParser.parseNodeFeature(line, $(featureDim), $(dataFormat)))

    def calculateMinMax(iterator: Iterator[(_, IntFloatVector)]): Iterator[Int] = {
      val min = VFactory.denseFloatVector(Array.fill($(featureDim))(Float.MaxValue))
      val max = VFactory.denseFloatVector(Array.fill($(featureDim))(Float.MinValue))

      while (iterator.hasNext) {
        val f = iterator.next()._2
        Ufuncs.imax(max, f)
        Ufuncs.imin(min, f)
      }

      model.updateMax(max)
      model.updateMin(min)
      Iterator(0)
    }

    def normalize(f: IntFloatVector, min: IntFloatVector, max: IntFloatVector): IntFloatVector = {
      f.getStorage match {
        case sorted: IntFloatSortedVectorStorage =>
          val indices = sorted.getIndices
          val values = sorted.getValues
          var j = 0
          while (j < indices.length) {
            val range = max.get(indices(j)) - min.get(indices(j))
            if (range > 0)
              values(j) = (values(j) - min.get(indices(j))) / range
            else
              values(j) = 0
            j += 1
          }
        case dense: IntFloatDenseVectorStorage =>
          val values = dense.getValues
          var j = 0
          while (j < values.length) {
            val range = max.get(j) - min.get(j)
            if (range > 0)
              values(j) = (values(j) - min.get(j)) / range
            else
              values(j) = 0
            j += 1
          }
      }
      f
    }

    def featureToString(key: Long, f: IntFloatVector): String = {
      f.getStorage match {
        case sorted: IntFloatSortedVectorStorage =>
          val indices = sorted.getIndices
          val values = sorted.getValues
          val ff = indices.zip(values).map(kv => s"${kv._1}:${kv._2}")
          s"$key ${ff.mkString(" ")}"
        case dense: IntFloatDenseVectorStorage =>
          val values = dense.getValues
          s"$key ${values.mkString(" ")}"
      }
    }

    def normalizePart(iterator: Iterator[(JLong, IntFloatVector)]): Iterator[(Long, IntFloatVector)] = {
      val max = model.readMax()
      val min = model.readMin()
      iterator.map(f => (f._1.toLong, normalize(f._2, min, max)))
    }

    features.mapPartitions(calculateMinMax).reduce(_ + _)
    val max = model.readMax().getStorage.getValues
    val min = model.readMin().getStorage.getValues

    val str = min.zip(max).map(f => s"${f._1}:${f._2}")
    println(str.mkString(" "))

    val output = features.mapPartitions(normalizePart)
      .map(f => featureToString(f._1, f._2))
      .map(f => Row.fromSeq(Seq[Any](f)))

    dataset.sparkSession.createDataFrame(output, transformSchema(dataset.schema))
  }

  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(
      StructField(s"feature", StringType, nullable = false)
    ))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
