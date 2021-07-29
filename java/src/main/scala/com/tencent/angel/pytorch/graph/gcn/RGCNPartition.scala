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

import java.util.{HashMap => JHashMap, Map => JMap}

import com.tencent.angel.exception.AngelException
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

class RGCNPartition(index: Int,
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    trainIdx: Array[Int],
                    trainLabels: Array[Array[Float]],
                    testIdx: Array[Int],
                    testLabels: Array[Array[Float]],
                    torchModelPath: String,
                    useSecondOrder: Boolean) extends
  GCNPartition(index, keys, indptr, neighbors, trainIdx, trainLabels,
    testIdx, testLabels, torchModelPath, useSecondOrder) {


  override
  def makeParams(batchIdx: Array[Int],
                 numSample: Int,
                 featureDim: Int,
                 model: GNNPSModel,
                 isTraining: Boolean,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): JMap[String, Object] = {
    val batchKeys = new LongOpenHashSet()
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val edgeTypes = new LongArrayList()

    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    val (first, firstTypes, second, secondTypes) = MakeEdgeIndex.makeEdgeIndex(batchIdx,
      keys, indptr, neighbors, types, srcs, dsts,
      edgeTypes, batchKeys, index, numSample, model)

    val params = new JHashMap[String, Object]()
    val (x, batchIds, fieldIds) = MakeSparseBiFeature.makeFeatures(index, featureDim, model, -1, params, fieldNum, fieldMultiHot)

    params.put("first_edge_index", first)
    params.put("first_edge_type", firstTypes)
    if (useSecondOrder) {
      params.put("second_edge_index", second)
      params.put("second_edge_type", secondTypes)
    }
    params.put("x", x)
    if (fieldNum > 0) {
      params.put("batch_ids", batchIds)
      params.put("field_ids", fieldIds)
    }
    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(featureDim))
    params

  }

  override
  def makeParams(nodes: Array[Long],
                 featureDim: Int,
                 model: GNNPSModel,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): JMap[String, Object] = {
    throw new AngelException("cannot determine edge types for nodes without edges")
  }

}