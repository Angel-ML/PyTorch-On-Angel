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
package com.tencent.angel.pytorch.examples.feature

import com.tencent.angel.spark.ml.core.ArgsUtil
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{FloatType, LongType, StringType, StructField, StructType}

object ExtractNodesWithLabels {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val featureInput = params.getOrElse("featureInput", "")
    val labelInput = params.getOrElse("labelInput", "")
    val output = params.getOrElse("output", "")

    val ss = SparkSession.builder().getOrCreate()
    val schema1 = StructType(Seq(
      StructField("node", LongType),
      StructField("label", FloatType)
    ))

    val labelDF = ss.read
      .option("header", "false")
      .option("sep", " ")
      .schema(schema1)
      .csv(labelInput)

    val schema2 = StructType(Seq(
      StructField("node", LongType),
      StructField("feature", StringType)
    ))

    val featureDF = ss.read
      .option("header", "false")
      .option("sep", "\t")
      .schema(schema2)
      .csv(featureInput)

    val labels = labelDF.select("node", "label").rdd
      .map(r => (r.getLong(0), r.getFloat(1))).collectAsMap()

    val bcLabels = labelDF.sparkSession.sparkContext.broadcast(labels)
    featureDF.select("node", "feature").rdd
      .map(r => (r.getLong(0), r.getString(1)))
      .filter(f => bcLabels.value.contains(f._1))
      .map(f => (bcLabels.value.get(f._1).get, f._2))
      .map(f => s"${f._1} ${f._2}")
      .repartition(20)
      .saveAsTextFile(output)
  }
}
