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

import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.floats.FloatArrayList
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

private[gcn]
class DGIPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   weights: Array[Float],
                   torchModelPath: String,
                   useSecondOrder: Boolean,
                   dataFormat: String) extends
  GNNPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder) {

  override def makeParams(nodes: Array[Long],
                          featureDim: Int,
                          model: GNNPSModel,
                          fieldNum: Int,
                          fieldMultiHot: Boolean): JMap[String, Object] = {
    val index = new Long2IntOpenHashMap()
    for (node <- nodes)
      index.put(node, index.size())
    val first = MakeEdgeIndex.makeEdgeIndex(nodes, index)
    val params = new JHashMap[String, Object]()
    val (x, batchIds, fieldIds) = MakeSparseBiFeature.makeFeatures(index, featureDim, model, -1, params, fieldNum, fieldMultiHot)

    params.put("first_edge_index", first)
    if (weights != null) {
      val edgeWeights = new Array[Float](first.length / 2)
      (0 until first.length / 2).foreach(i => edgeWeights(i) = 0.5f)
      params.put("first_edge_weight", edgeWeights)
    }
    if (useSecondOrder)
      params.put("second_edge_index", first)
    params.put("x", x)
    if (fieldNum > 0) {
      params.put("batch_ids", batchIds)
      params.put("field_ids", fieldIds)
    }
    params.put("batch_size", new Integer(nodes.length))
    params.put("feature_dim", new Integer(featureDim))
    params
  }

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 fieldNum: Int,
                 fieldMultiHot: Boolean,
                 trainRatio: Float): (Double, Long, Int) = {
    val batchIterator = sampleTrainData(keys.indices, trainRatio).sliding(batchSize, batchSize)
    var lossSum = 0.0
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0
    while (batchIterator.hasNext) {
      val batch = batchIterator.next().toArray
      val loss = trainBatch(batch, model, featureDim, optim, numSample, torch, fieldNum, fieldMultiHot)
      lossSum += loss * batch.length
      numStep += 1
    }

    TorchModel.put(torch)
    (lossSum, keys.length, numStep)
  }

  def trainBatch(batchIdx: Array[Int],
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 torch: TorchModel,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): Double = {
    val param = makeParams(batchIdx, numSample, featureDim, model, true, fieldNum, fieldMultiHot)
    val weights = model.readWeights()
    param.put("weights", weights)
    val loss = torch.gcnBackward(param, fieldNum > 0)
    model.step(weights, optim)

    // update grad for embedding of sparse
    if (fieldNum > 0) {
      //check if the grad really replaced the pulledUEmbedding
      val pulledEmbedding = param.get("pulledEmbedding").asInstanceOf[Array[Vector]]
      val feats = param.get("feats").asInstanceOf[Array[Int]]
      val embeddingInput = param.get("pos_x").asInstanceOf[Array[Float]]
      val embeddingDim = pulledEmbedding.length
      makeEmbeddingGrad(embeddingInput, pulledEmbedding, feats, embeddingDim)
      model.asInstanceOf[SparseGNNPSModel].updateEmbedding(pulledEmbedding, optim)
    }
    loss
  }

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
    val edgeWeights = new FloatArrayList()

    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    val (first, second) = if (weights == null) {
      MakeEdgeIndex.makeEdgeIndex(batchIdx,
        keys, indptr, neighbors, srcs, dsts, batchKeys,
        index, numSample, model, useSecondOrder)
    } else {
      MakeEdgeIndex.makeEdgeIndex(batchIdx,
        keys, indptr, neighbors, weights, srcs, dsts, edgeWeights, batchKeys,
        index, numSample, model, useSecondOrder)
    }

    val params = new JHashMap[String, Object]()
    val (posx, batchIds, fieldIds) = MakeSparseBiFeature.makeFeatures(index, featureDim, model, -1, params, fieldNum, fieldMultiHot)
    val (negx, negBatchIds, negFieldIds) = if (isTraining)
      MakeSparseBiFeature.sampleFeatures(index.size(), featureDim, model, -1, dataFormat, params, fieldNum, fieldMultiHot)
    else (null, null, null)
    if (isTraining && !fieldMultiHot)
      assert(posx.length == negx.length)


    params.put("first_edge_index", first)
    if (weights != null) params.put("first_edge_weight", edgeWeights.toFloatArray)
    if (second != null)
      params.put("second_edge_index", second)
    if (isTraining) {
      params.put("pos_x", posx)
      params.put("neg_x", negx)
      if (fieldNum > 0) {
        params.put("batch_ids", batchIds)
        params.put("field_ids", fieldIds)
        params.put("neg_batch_ids", negBatchIds)
        params.put("neg_field_ids", negFieldIds)
      }
    } else {
      params.put("x", posx)
      if (fieldNum > 0) {
        params.put("batch_ids", batchIds)
        params.put("field_ids", fieldIds)
      }
    }
    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(featureDim))
    params
  }

}
