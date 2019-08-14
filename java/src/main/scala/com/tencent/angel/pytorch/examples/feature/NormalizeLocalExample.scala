package com.tencent.angel.pytorch.examples.feature

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.pytorch.feature.normalize.MinMaxNormalize
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object NormalizeLocalExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val featureInput = params.getOrElse("featureInput", "")
    val featureOutput = params.getOrElse("featureOutput", "")
    val format = params.getOrElse("format", "dense")
    val featureDim = params.getOrElse("featureDim", "0").toInt

    start()

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

    stop()
  }

  def start(): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("normalize")
    conf.set(AngelConf.ANGEL_PSAGENT_UPDATE_SPLIT_ADAPTION_ENABLE, "false")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("cp")
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}
