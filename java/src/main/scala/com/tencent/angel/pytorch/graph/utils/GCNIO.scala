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
