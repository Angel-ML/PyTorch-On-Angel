package com.tencent.angel.pytorch.examples.feature

import com.tencent.angel.pytorch.feature.normalize.MinMaxNormalize
import com.tencent.angel.spark.ml.core.ArgsUtil
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{SaveMode, SparkSession}

object NormalizeExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val featureInput = params.getOrElse("featureInput", "")
    val featureOutput = params.getOrElse("featureOutput", "")
    val format = params.getOrElse("format", "dense")
    val featureDim = params.getOrElse("featureDim", "0").toInt

    val ss = SparkSession.builder().getOrCreate()
    val schema = StructType(Seq(StructField("feature", StringType)))
    val input = ss.read
      .option("header", "false")
      .option("sep", "\t")
      .schema(schema)
      .csv(featureInput)

    val normalize = new MinMaxNormalize()
    normalize.setDataFormat(format)
    normalize.setFeatureDim(featureDim)

    val df = normalize.transform(input)
    df.write
      .mode(SaveMode.Overwrite)
      .option("header", "false")
      .option("delimiter", "\t")
      .csv(featureOutput)
  }

}
