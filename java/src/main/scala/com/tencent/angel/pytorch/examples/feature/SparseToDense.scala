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

import com.tencent.angel.ml.math2.storage.IntFloatSortedVectorStorage
import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.spark.ml.core.ArgsUtil
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}

object SparseToDense {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val featureInput = params.getOrElse("featureInput", "")
    val featureOutput = params.getOrElse("featureOutput", "")
    val featureDim = params.getOrElse("featureDim", "-1").toInt

    assert(featureDim > 0)

    val ss = SparkSession.builder().getOrCreate()
    val schema = StructType(Seq(
      StructField("node", LongType),
      StructField("feature", StringType)
    ))

    val input = ss.read
      .option("header", "false")
      .option("sep", "\t")
      .schema(schema)
      .csv(featureInput)

    input.select("node", "feature")
      .rdd.map(row => (row.getLong(0), row.getString(1)))
      .map(f => (f._1, SampleParser.parseSparseIntFloat(f._2, featureDim)))
      .map { case (node, f) =>
        val str = f.getStorage match {
          case sorted: IntFloatSortedVectorStorage =>
            val indices = sorted.getIndices
            val values = sorted.getValues
            val dense = new Array[Float](featureDim)
            for (idx <- indices.indices)
              dense(indices(idx)) = values(idx)
            dense.mkString(" ")
        }
        s"$node\t$str"
      }.saveAsTextFile(featureOutput)
  }

}
