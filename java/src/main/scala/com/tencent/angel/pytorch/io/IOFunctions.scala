/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
package com.tencent.angel.pytorch.io

import com.tencent.angel.exception.AngelException
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}

object IOFunctions {

  def loadString(input: String, index: Int = 0): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()
    val schema = StructType(Seq(
      StructField("example", StringType, nullable = false)
    ))
    ss.read
      .option("sep", "\t")
      .option("header", "false")
      .schema(schema)
      .csv(input)
  }

  def loadFeature(input: String,
                  nodeIndex: Int = 0, featureIndex: Int = 1,
                  sep: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()
    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("feature", StringType, nullable = false)
    ))
    val df = ss.read
      .option("sep", sep)
      .option("header", "false")
      .schema(schema)
      .csv(input)
    df.persist()
    if (df.rdd.filter(row => row.get(0) != null).count() == 0) throw new AngelException("The feature format is incorrect, please check!!!")
    df
  }

  def loadLabel(input: String, nodeIndex: Int = 0,
                labelIndex: Int = 1, seq: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()
    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", FloatType, nullable = false)
    ))
    val df = ss.read
      .option("sep", seq)
      .option("header", "false")
      .schema(schema)
      .csv(input)
    df.persist()
    if (df.rdd.filter(row => row.get(0) != null).count() == 0) throw new AngelException("The label format is incorrect, please check!!!")
    df
  }

  def loadMultiLabel(input: String, index: Int = 0, sep: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()
    val schema = StructType(Seq(
      StructField("label", StringType, nullable = false)
    ))
    val df= ss.read
      .option("sep", sep)
      .option("header", "false")
      .schema(schema)
      .csv(input)
    df.persist()
    if (df.rdd.filter(row => row.get(0) != null).count() == 0) throw new AngelException("The multi-label format is incorrect, please check!!!")
    df
  }

  def loadEdge(input: String, isTyped: Boolean,
               srcIndex: Int = 0, dstIndex: Int = 1, typeIndex: Int = 2,
               sep: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()
    val schema = if (isTyped) {
      StructType(Seq(
        StructField("src", LongType, nullable = false),
        StructField("dst", LongType, nullable = false),
        StructField("type", IntegerType, nullable = false)
      ))
    } else {
      StructType(Seq(
        StructField("src", LongType, nullable = false),
        StructField("dst", LongType, nullable = false)
      ))
    }
    val df = ss.read
      .option("sep", sep)
      .option("header", "false")
      .schema(schema)
      .csv(input)
    df.persist()
    if (df.rdd.filter(row => row.get(0) != null).count() == 0) throw new AngelException("The edge format is incorrect, please check!!!")
    df
  }

  def loadEdgeFeature(input: String, isTyped: Boolean = false,
                      srcIndex: Int = 0, dstIndex: Int = 1, featureIndex: Int = 2,
                      sep: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()
    val schema = StructType(Seq(
      StructField("src", LongType, nullable = false),
      StructField("dst", LongType, nullable = false),
      StructField("feature", StringType, nullable = false)
    ))
    val df = ss.read
      .option("sep", sep)
      .option("header", "false")
      .schema(schema)
      .csv(input)
    df.persist()
    if (df.rdd.filter(row => row.get(0) != null).count() == 0) throw new AngelException("The edge feature format is incorrect, please check!!!")
    df
  }
}
