package com.tencent.angel.pytorch.graph.utils

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}

object GCNIO {

  def loadFeature(input: String,
                  nodeIndex: Int = 0, featureIndex: Int = 1,
                  sep: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()


    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("feature", StringType, nullable = false)
    ))
    ss.read
      .option("sep", sep)
      .option("header", "false")
      .schema(schema)
      .csv(input)
  }

  def loadLabel(input: String, nodeIndex: Int = 0,
                labelIndex: Int = 1, seq: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", FloatType, nullable = false)
    ))
    ss.read
      .option("sep", seq)
      .option("header", "false")
      .schema(schema)
      .csv(input)
  }


}
