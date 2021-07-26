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

import java.util
import java.util.{HashMap => JHashMap, Map => JMap}
import com.tencent.angel.ml.math2.vector.IntFloatVector
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

class EdgePropGCNPartition(index: Int,
                           keys: Array[Long],
                           indptr: Array[Int],
                           neighbors: Array[Long],
                           edgeDim: Int,
                           features: Array[IntFloatVector],
                           trainIdx: Array[Int],
                           trainLabels: Array[Array[Float]],
                           testIdx: Array[Int],
                           testLabels: Array[Array[Float]],
                           torchModelPath: String,
                           useSecondOrder: Boolean = true) extends
  GCNPartition(index, keys, indptr, neighbors, trainIdx, trainLabels, testIdx, testLabels, torchModelPath, useSecondOrder) {

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
    val edgeFeatures = new util.ArrayList[IntFloatVector]()

    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    val (first, second, firstFeature, secondFeature) = MakeEdgeIndex.makeEdgeIndex(batchIdx,
      keys, indptr, neighbors, edgeDim, features, srcs, dsts, edgeFeatures,
      batchKeys, index, numSample, model, useSecondOrder)

    val params = new JHashMap[String, Object]()
    val (x, batchIds, fieldIds) = MakeSparseBiFeature.makeFeatures(index, featureDim, model, -1, params, fieldNum,
      fieldMultiHot)

    params.put("first_edge_index", first)
    params.put("first_edge_feature", firstFeature)
    if (useSecondOrder) {
      params.put("second_edge_index", second)
      params.put("second_edge_feature", secondFeature)
    }
    params.put("x", x)
    if (fieldNum > 0) {
      params.put("batch_ids", batchIds)
      params.put("field_ids", fieldIds)
    }
    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(featureDim))
    params.put("edge_dim", new Integer(edgeDim))
    params
  }

  override
  def makeParams(nodes: Array[Long],
                 featureDim: Int,
                 model: GNNPSModel,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): JMap[String, Object] = {
    val index = new Long2IntOpenHashMap()
    for (node <- nodes)
      index.put(node, index.size())
    val first = MakeEdgeIndex.makeEdgeIndex(nodes, index)
    val size = first.length / 2 * edgeDim
    val firstFeature = new Array[Float](size)
    val params = new JHashMap[String, Object]()
    val (x, batchIds, fieldIds) = MakeSparseBiFeature.makeFeatures(index, featureDim, model, -1, params, fieldNum, fieldMultiHot)

    params.put("first_edge_index", first)
    params.put("first_edge_feature", firstFeature)
    if (useSecondOrder) {
      params.put("second_edge_index", first)
      params.put("second_edge_feature", firstFeature)
    }
    params.put("x", x)
    if (fieldNum > 0) {
      params.put("batch_ids", batchIds)
      params.put("field_ids", fieldIds)
    }
    params.put("batch_size", new Integer(nodes.length))
    params.put("feature_dim", new Integer(featureDim))
    params.put("edge_dim", new Integer(edgeDim))
    params
  }
}