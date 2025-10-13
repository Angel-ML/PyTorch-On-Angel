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
