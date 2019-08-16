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
package com.tencent.angel.pytorch.examples

import com.tencent.angel.spark.ml.core.ArgsUtil
import it.unimi.dsi.fastutil.ints.IntOpenHashSet
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

object SubFeature {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val featureInput = params.getOrElse("featureInput", "")
    val labelInput = params.getOrElse("labelInput", "")
    val output = params.getOrElse("output", "")

    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val features = sc.textFile(featureInput)
    val labels = sc.textFile(labelInput).map(f => f.stripLineEnd.split(" ")(0).toInt).collect()

    def extract(iterator: Iterator[String]): Iterator[String] = {
      val set = new IntOpenHashSet()
      labels.foreach(f => set.add(f))
      val results = new ArrayBuffer[String]()
      while (iterator.hasNext) {
        val line = iterator.next()
        val parts = line.stripLineEnd.split(" ")
        val key = parts(0).toInt
        if (labels.contains(key))
          results.append(line)
      }
      results.iterator
    }

    features.mapPartitions(extract).saveAsTextFile(output)

  }

}
