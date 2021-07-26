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
package com.tencent.angel.pytorch.graph.gcn.hetAttention

import java.util.{HashMap => JHashMap, Map => JMap}
import com.tencent.angel.pytorch.graph.gcn._
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList}

class HANPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   types: Array[Int],
                   userTrainIdx: Array[Int],
                   userTrainLabels:  Array[Array[Float]],
                   userTestIdx: Array[Int],
                   userTestLabels:  Array[Array[Float]],
                   torchModelPath: String,
                   itemTypes: Int,
                   useSecondOrder: Boolean=true)
  extends GCNPartition(index, keys, indptr, neighbors, userTrainIdx, userTrainLabels, userTestIdx,
    userTestLabels, torchModelPath, useSecondOrder) {

  override def makeParams(batchIdx: Array[Int],
                          numSample: Int,
                          userFeatureDim: Int,
                          model: GNNPSModel,
                          isTraining: Boolean,
                          fieldNum: Int,
                          fieldMultiHot: Boolean): JMap[String, Object] = {
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val edgeTypes = new LongArrayList()

    val index = new Long2IntOpenHashMap()
    for (id <- batchIdx) {
      index.put(keys(id), index.size())
    }

    val (second, secondTypes) = MakeEdgeIndex.makeHANEdgeIndex(batchIdx,
      keys, indptr, neighbors, types, srcs, dsts, edgeTypes,
      index, numSample, model, itemTypes)

    val params = new JHashMap[String, Object]()
    val (x, batchIds, fieldIds) = MakeSparseBiFeature.makeFeatures(index, userFeatureDim, model, -1, params, fieldNum,
      fieldMultiHot)

    params.put("second_edge_index", second)
    params.put("second_edge_type", secondTypes)
    params.put("x", x)
    if (fieldNum > 0) {
      params.put("batch_ids", batchIds)
      params.put("field_ids", fieldIds)
    }
    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(userFeatureDim))
    params
  }
}