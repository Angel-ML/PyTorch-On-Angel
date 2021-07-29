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

import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList}

class HGATPartition(index: Int,
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    torchModelPath: String,
                    useSecondOrder: Boolean,
                    dataFormat: String) extends
  BiSAGEPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder,dataFormat) {


  override def makeParams(batchIdx: Array[Int],
                          userNumSample: Int,
                          itemNumSample: Int,
                          userFeatureDim: Int,
                          itemFeatureDim: Int,
                          model: BiSAGEPSModel,
                          graphType: Int,
                          isTraining: Boolean,
                          userFieldNum: Int,
                          itemFieldNum: Int,
                          fieldMultiHot: Boolean): JMap[String, Object] = {
    val srcIndex = new Long2IntOpenHashMap()
    val dstIndex = new Long2IntOpenHashMap()
    var srcs = new LongArrayList()
    var dsts = new LongArrayList()
    var srcs_ = new LongArrayList()
    var dsts_ = new LongArrayList()
    for (idx <- batchIdx)
      srcIndex.put(keys(idx), srcIndex.size())

    val edgesItems = if (isTraining)
      MakeBiEdgeIndex.makeAllEdgeIndex(batchIdx, keys, indptr, neighbors, srcs, dsts, srcIndex, dstIndex) else null


    // sample edges u-i-u, i-u-i
    val sampleNums_first = if (graphType==0) userNumSample else itemNumSample
    val sampleNums_second = if (graphType==0) itemNumSample else userNumSample

    val (first, firstKeys) = MakeBiEdgeIndex.makeFirstOrderEdgeIndex(
      batchIdx, keys, indptr, neighbors, srcs, dsts, srcIndex, dstIndex, sampleNums_first)

    val (second, _) = MakeBiEdgeIndex.makeSecondOrderEdgeIndex(firstKeys, srcs_, dsts_,
      dstIndex, srcIndex, model, sampleNums_second, 1 - graphType)

    if (isTraining) {
      srcs = new LongArrayList()
      dsts = new LongArrayList()
      srcs_ = new LongArrayList()
      dsts_ = new LongArrayList()
    }

    // i-u-i
    val (firstI, firstUKeys) = if (isTraining)
      MakeBiEdgeIndex.makeSecondOrderEdgeIndex(edgesItems, srcs_, dsts_,
        dstIndex, srcIndex, model, sampleNums_second, 1)
    else (null, null)

    val (secondU, _) = if (isTraining)
      MakeBiEdgeIndex.makeSecondOrderEdgeIndex(firstUKeys, srcs, dsts, srcIndex, dstIndex, model, sampleNums_first, graphType)
    else (null, null)


    val params = new JHashMap[String, Object]()

    val (pos_u, uBatchIds, uFieldIds) = if (graphType == 0)
      MakeSparseBiFeature.makeFeatures(srcIndex, userFeatureDim, model, 0, params, userFieldNum, fieldMultiHot)
    else
      MakeSparseBiFeature.makeFeatures(dstIndex, userFeatureDim, model, 0, params, userFieldNum, fieldMultiHot)

    val (pos_i, iBatchIds, iFieldIds) = if (itemFeatureDim > 0) {
      if (graphType == 0)
        MakeSparseBiFeature.makeFeatures(dstIndex, itemFeatureDim, model, 1, params, itemFieldNum, fieldMultiHot)
      else
        MakeSparseBiFeature.makeFeatures(srcIndex, itemFeatureDim, model, 1, params, itemFieldNum, fieldMultiHot)
    } else (null, null, null)


    if(graphType == 0){
      // sample u-i-u
      params.put("first_u_edge_index", first)
      params.put("first_i_edge_index", second)

      if (isTraining) { //sample i-u-i
        params.put("second_u_edge_index", firstI)
        params.put("second_i_edge_index", secondU)

      }
    }
    else{
      // sample i-u-i-u
      params.put("first_u_edge_index", first)
      params.put("first_i_edge_index", second)
    }
    if (isTraining) {
      params.put("pos_u", pos_u)
      if (userFieldNum > 0) {
        params.put("u_batch_ids", uBatchIds)
        params.put("u_field_ids", uFieldIds)
      }
      if (itemFeatureDim > 0){
        params.put("pos_i", pos_i)
        if (itemFieldNum > 0) {
          params.put("i_batch_ids", iBatchIds)
          params.put("i_field_ids", iFieldIds)
        }
      }
    }else{
      params.put("u", pos_u)
      if (itemFeatureDim > 0){
        params.put("i", pos_i)
      }

      if (userFieldNum > 0) {
        params.put("u_batch_ids", uBatchIds)
        params.put("u_field_ids", uFieldIds)
      }
      if (itemFeatureDim > 0){
        if (itemFieldNum > 0) {
          params.put("i_batch_ids", iBatchIds)
          params.put("i_field_ids", iFieldIds)
        }
      }
    }

    params.put("batch_size", new Integer(srcIndex.size()))
    params.put("feature_dim", new Integer(userFeatureDim))
    params.put("user_feature_dim", new Integer(userFeatureDim))
    if (itemFeatureDim > 0){
      params.put("item_feature_dim", new Integer(itemFeatureDim))
    }

    params
  }
}