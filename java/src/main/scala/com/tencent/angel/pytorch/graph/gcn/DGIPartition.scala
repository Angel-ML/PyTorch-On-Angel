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

import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

private[gcn]
class DGIPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   torchModelPath: String,
                   useSecondOrder: Boolean) extends
  GNNPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder) {

  override
  def makeParams(batchIdx: Array[Int],
                 numSample: Int,
                 featureDim: Int,
                 model: GNNPSModel,
                 isTraining: Boolean = true): JMap[String, Object] = {
    val batchKeys = new LongOpenHashSet()
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()

    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx,
      keys, indptr, neighbors, srcs, dsts, batchKeys,
      index, numSample, model, useSecondOrder)

    val posx = MakeFeature.makeFeatures(index, featureDim, model)
    val negx = if (isTraining) MakeFeature.sampleFeatures(index.size(), featureDim, model) else null
    if (isTraining)
      assert(posx.length == negx.length)

    val params = new JHashMap[String, Object]()
    params.put("first_edge_index", first)
    if (second != null)
      params.put("second_edge_index", second)
    if (isTraining) {
      params.put("pos_x", posx)
      params.put("neg_x", negx)
    } else
      params.put("x", posx)
    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(featureDim))
    params
  }

  override
  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int): (Double, Long, Int) = {
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    var lossSum = 0.0
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0
    while (batchIterator.hasNext) {
      val batch = batchIterator.next().toArray
      val loss = trainBatch(batch, model, featureDim, optim, numSample, torch)
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
                 torch: TorchModel): Double = {

    val param = makeParams(batchIdx, numSample, featureDim, model)
    val weights = model.readWeights()
    param.put("weights", weights)
    val loss = torch.gcnBackward(param)
    model.step(weights, optim)
    loss
  }

}
