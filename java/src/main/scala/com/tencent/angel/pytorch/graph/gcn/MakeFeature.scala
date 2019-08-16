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
package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.ml.math2.storage.{IntFloatDenseVectorStorage, IntFloatSortedVectorStorage}
import com.tencent.angel.ml.math2.vector.IntFloatVector
import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap

object MakeFeature {

  def makeFeatures(index: Long2IntOpenHashMap, featureDim: Int, model: GNNPSModel): Array[Float] = {
    val size = index.size()
    val x = new Array[Float](size * featureDim)
    val keys = index.keySet().toLongArray
    val features = model.getFeatures(keys)
    //    assert(features.size() == keys.length)
    val it = features.long2ObjectEntrySet().fastIterator()
    while (it.hasNext) {
      val entry = it.next()
      val (node, f) = (entry.getLongKey, entry.getValue)
      val start = index.get(node) * featureDim
      makeFeature(start, f, x)
    }
    x
  }

  def makeFeature(start: Int, f: IntFloatVector, x: Array[Float]): Unit = {
    f.getStorage match {
      case sorted: IntFloatSortedVectorStorage =>
        val indices = sorted.getIndices
        val values = sorted.getValues
        var j = 0
        while (j < indices.length) {
          x(start + indices(j)) = values(j)
          j += 1
        }
      case dense: IntFloatDenseVectorStorage =>
        val values = dense.getValues
        var j = 0
        while (j < values.length) {
          x(start + j) = values(j)
          j += 1
        }
    }
  }

  def sampleFeatures(size: Int, featureDim: Int, model: GNNPSModel): Array[Float] = {
    val x = new Array[Float](size * featureDim)
    val features = model.sampleFeatures(size)
    for (idx <- 0 until size)
      makeFeature(idx * featureDim, features(idx), x)
    x
  }


}
