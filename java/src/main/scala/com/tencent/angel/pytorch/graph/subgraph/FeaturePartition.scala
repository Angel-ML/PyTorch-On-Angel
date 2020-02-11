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
package com.tencent.angel.pytorch.graph.subgraph

import it.unimi.dsi.fastutil.longs.LongArrayList

import scala.collection.mutable.ArrayBuffer

private[subgraph]
class FeaturePartition(index: Int,
                       keys: Array[Long],
                       features: Array[String]) extends Serializable {

  def sample(model: SubGraphPSModel): Iterator[(Long, String)] = {
    val flags = model.readNodes(keys.clone())
    val results = new ArrayBuffer[(Long, String)]()
    for (idx <- keys.indices) {
      if (flags.get(keys(idx)) > 0) {
        results.append((keys(idx), features(idx)))
      }
    }
    results.iterator
  }
}

private[subgraph]
object FeaturePartition {
  //  def apply(index: Int, iterator: Iterator[(Long, String)]): FeaturePartition = {
  //    val fs = iterator.toArray
  //    val keys = fs.map(f => f._1)
  //    val features = fs.map(f => f._2)
  //    new FeaturePartition(index, keys, features)
  //  }

  def apply(index: Int, iterator: Iterator[String]): FeaturePartition = {
    val keys = new LongArrayList()
    val features = new ArrayBuffer[String]()
    while (iterator.hasNext) {
      val line = iterator.next()
      val parts = line.stripLineEnd.split(" ")

      keys.add(parts(0).toLong)
      features.append(parts.tail.mkString(" "))
    }
    new FeaturePartition(index, keys.toLongArray, features.toArray)
  }
}
