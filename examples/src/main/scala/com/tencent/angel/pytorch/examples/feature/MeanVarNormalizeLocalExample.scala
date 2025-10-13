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

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.pytorch.feature.normalize.MeanVarNormalize
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object MeanVarNormalizeLocalExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val featureInput = params.getOrElse("featureInput", "")
    val featureOutput = params.getOrElse("featureOutput", "")
    val format = params.getOrElse("format", "dense")
    val featureDim = params.getOrElse("featureDim", "-1").toInt

    assert(featureDim > 0)

    start()

    val ss = SparkSession.builder().getOrCreate()
    val schema = StructType(Seq(
      StructField("node", LongType),
      StructField("feature", StringType)))

    val input = ss.read
      .option("header", "false")
      .option("sep", "\t")
      .schema(schema)
      .csv(featureInput)

    val mv = new MeanVarNormalize()
    mv.setDataFormat(format)
    mv.setFeatureDim(featureDim)

    val df = mv.transform(input)
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
