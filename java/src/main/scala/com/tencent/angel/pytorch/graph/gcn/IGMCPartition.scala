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
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList}

class IGMCPartition(index: Int,
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    userTrainIdx: Array[(Int, Int)],
                    userTrainLabels: Array[Int],
                    userTestIdx: Array[(Int, Int)],
                    userTestLabels: Array[Int],
                    keysI: Array[Long],
                    torchModelPath: String,
                    useSecondOrder: Boolean,
                    hasEdgeType: Boolean)
  extends GNNPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder) {

  def getTrainTestSize(): (Int, Int) = (userTrainIdx.length, userTestIdx.length)

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: BiSAGEPSModel,
                 userFeatureDim: Int,
                 itemFeatureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 graphType: Int,
                 isSharedSamples: Boolean): (Double, Long, Int, Array[(Float, Float)]) = {
    val batchIterator = userTrainIdx.zip(userTrainLabels).sliding(batchSize, batchSize)
    var lossSum = 0.0

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0

    val pairs = new Iterator[Array[(Float, Float)]] with Serializable {
      override def hasNext: Boolean = {
        batchIterator.hasNext
      }

      override def next: Array[(Float, Float)] = {
        val batch = batchIterator.next()
        val (loss, output) = trainBatch(batch, model, userFeatureDim, itemFeatureDim, optim, numSample, torch, graphType, isSharedSamples)
        lossSum += loss * batch.length
        numStep += 1
        output._1.zip(output._2)
      }
    }.flatMap(p => p).toArray

    TorchModel.put(torch)
    (lossSum, keys.length, numStep, pairs)
  }

  def trainBatch(batchIdx: Array[((Int, Int), Int)],
                 model: BiSAGEPSModel,
                 userFeatureDim: Int,
                 itemFeatureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 torch: TorchModel,
                 graphType: Int,
                 isSharedSamples: Boolean): (Double, (Array[Float], Array[Float])) = {

    val targets = new Array[Float](batchIdx.length)
    var k = 0
    for ((_, label) <- batchIdx) {
      targets(k) = label.toFloat
      k += 1
    }

    val params = makeParams(batchIdx.map(_._1), numSample, userFeatureDim, itemFeatureDim, model, graphType, true)
    var weights = model.readWeights()
    params.put("weights", weights)
    params.put("targets", targets)
    val loss = torch.gcnBackward(params, false)

    model.step(weights, optim)

    val output = if (isSharedSamples) {
      weights = model.readWeights()
      params.remove("weights")
      params.put("weights", weights)
      predictBatch(params, targets, torch)
    } else {
      (new Array[Float](0), new Array[Float](0))
    }


    (loss, output)
  }

  def predictBatch(params: JMap[String, Object],
                   targets: Array[Float],
                   torch: TorchModel): (Array[Float], Array[Float]) = {
    val outputs = torch.gcnPredict(params)
    outputs match {
      case f: Array[Float] => (targets, f)
      case l: Array[Long] => (targets, l.map(x => x.toFloat))
    }
  }

  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: BiSAGEPSModel,
                   userFeatureDim: Int,
                   itemFeatureDim: Int,
                   numSample: Int,
                   isTest: Boolean): Iterator[(Array[Float], Array[Float])] = {

    val zips = if (isTest) userTestIdx.zip(userTestLabels) else userTrainIdx.zip(userTrainLabels)
    val batchIterator = zips.sliding(batchSize, batchSize)
    val weights = model.readWeights()

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()

    new Iterator[(Array[Float], Array[Float])] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext)
          TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: (Array[Float], Array[Float]) = {
        val batch = batchIterator.next()
        predictBatch(batch, model, userFeatureDim, itemFeatureDim, numSample, weights, torch)
      }
    }
  }

  def predictBatch(batchIdx: Array[((Int, Int), Int)],
                   model: BiSAGEPSModel,
                   userFeatureDim: Int,
                   itemFeatureDim: Int,
                   numSample: Int,
                   weights: Array[Float],
                   torch: TorchModel): (Array[Float], Array[Float]) = {
    val targets = batchIdx.map(f => f._2.toFloat)
    val params = makeParams(batchIdx.map(f => f._1), numSample, userFeatureDim, itemFeatureDim, model, 0, false)
    params.put("weights", weights)
    val outputs = torch.gcnPredict(params)
    outputs match {
      case f: Array[Float] => (targets, f)
      case l: Array[Long] => (targets, l.map(x => x.toFloat))
    }
  }

  def genLabel(batchSize: Int,
               model: BiSAGEPSModel,
               userFeatureDim: Int,
               itemFeatureDim: Int,
               numSample: Int,
               numPartitions: Int,
               graphType: Int,
               taskType: String,
               parseAloneNodes: Boolean = true): Iterator[(Long, Long, String, String)] = {

    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val batchIterator = (userTrainIdx ++ userTestIdx)
      .sliding(batchSize, batchSize)
    val keysIterator = new Iterator[Array[(Long, Long, String, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, Long, String, String)] = {
        val batch = batchIterator.next()
        val outputs = genLabelsBatch(batch, model, userFeatureDim, itemFeatureDim, numSample,
          weights, torch, graphType, taskType)
        outputs.toArray
      }
    }

    keysIterator.flatMap(f => f.iterator)
  }

  def genLabelsBatch(batchIdx: Array[(Int, Int)],
                     model: BiSAGEPSModel,
                     userFeatureDim: Int,
                     itemFeatureDim: Int,
                     numSample: Int,
                     weights: Array[Float],
                     torch: TorchModel,
                     graphType: Int,
                     taskType: String): Iterator[(Long, Long, String, String)] = {


    val batchIds = batchIdx.map(idx => (keys(idx._1), keysI(idx._2)))
    val params = makeParams(batchIdx, numSample, userFeatureDim, itemFeatureDim, model, graphType, false)
    params.put("weights", weights)
    val outputs = torch.gcnPredict(params) match {
      case f: Array[Float] => f
      case f: Array[Long] => f.map(x => x.toFloat)
    }
    assert(outputs.length % batchIds.length == 0)
    val numLabels = outputs.length / batchIds.length
    if (taskType == "classification") {
      outputs.sliding(numLabels, numLabels)
        .zip(batchIds.iterator).map {
        case (p, key) =>
          val maxIndex =
            if (p.length == 1)
              if (p(0) >= 0.5) 1L
              else 0L
            else p.zipWithIndex.maxBy(_._1)._2
          (key._1, key._2, maxIndex.toString, p.mkString(","))
      }
    } else {
      outputs.sliding(1, 1)
        .zip(batchIds.iterator).map {
        case (p, key) =>
          val maxIndex = p(0).toString
          (key._1, key._2, maxIndex, p.mkString(","))
      }
    }
  }

  def genEmbedding(batchSize: Int,
                   model: BiSAGEPSModel,
                   userFeatureDim: Int,
                   itemFeatureDim: Int,
                   numSample: Int,
                   numPartitions: Int,
                   graphType: Int): Iterator[(Long, Long, String)] = {

    val batchIterator = (userTrainIdx ++ userTestIdx).sliding(batchSize, batchSize)
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()

    val keyIterator = new Iterator[Array[(Long, Long, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, Long, String)] = {
        val batch = batchIterator.next()
        val output = genEmbeddingBatch(batch, model, userFeatureDim, itemFeatureDim,
          numSample, weights, torch, graphType)
        output.toArray
      }
    }

    keyIterator.flatMap(f => f.iterator)
  }

  def genEmbeddingBatch(batchIdx: Array[(Int, Int)],
                        model: BiSAGEPSModel,
                        userFeatureDim: Int,
                        itemFeatureDim: Int,
                        numSample: Int,
                        weights: Array[Float],
                        torch: TorchModel,
                        graphType: Int): Iterator[(Long, Long, String)] = {

    val batchIds = batchIdx.map(idx => (keys(idx._1), keysI(idx._2)))
    val params = makeParams(batchIdx, numSample, userFeatureDim, itemFeatureDim, model, graphType, false)
    params.put("weights", weights)

    val outputs = torch.gcnEmbedding(params)
    assert(outputs.length % batchIds.length == 0)
    val numLabels = outputs.length / batchIds.length

    outputs.sliding(numLabels, numLabels)
      .zip(batchIds.iterator).map {
      case (p, key) =>
        (key._1, key._2, p.mkString(","))
    }
  }

  def makeParams(batchIdx: Array[(Int, Int)],
                 numSample: Int,
                 userFeatureDim: Int,
                 itemFeatureDim: Int,
                 model: BiSAGEPSModel,
                 graphType: Int,
                 isTraining: Boolean): JMap[String, Object] = {
    val srcIndex = new Long2IntOpenHashMap()
    val dstIndex = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val edgeTypes = new IntArrayList()

    for (idx <- batchIdx) {
      if (!srcIndex.containsKey(keys(idx._1))) srcIndex.put(keys(idx._1), srcIndex.size())
      if (!dstIndex.containsKey(keysI(idx._2))) dstIndex.put(keysI(idx._2), dstIndex.size())
    }

    //sample edges(neighbors)
    val (first, eTypes) = MakeBiEdgeIndex.makeEdgeIndex(batchIdx, keys,
      keysI, indptr, neighbors, types, srcs, dsts, edgeTypes, srcIndex, dstIndex, numSample,
      model, useSecondOrder, graphType, hasEdgeType)

    val pos_u = if (userFeatureDim > 0) MakeBiFeature.makeFeatures(srcIndex, userFeatureDim, model, 0) else null
    val pos_i = if (itemFeatureDim > 0) MakeBiFeature.makeFeatures(dstIndex, itemFeatureDim, model, 1) else null

    val params = new JHashMap[String, Object]()
    val size = batchIdx.length
    val edges = new Array[Long](size * 2)

    batchIdx.indices.foreach { i =>
      edges(i) = srcIndex.get(keys(batchIdx(i)._1))
      edges(i + size) = dstIndex.get(keysI(batchIdx(i)._2))
    }

    params.put("labeled_edge_index", edges)

    // sample u-i-u
    params.put("edge_index", first)
    if (hasEdgeType && eTypes != null && eTypes.length != 0) {
      params.put("edge_type", eTypes)
    }

    if (userFeatureDim > 0)
      params.put("u", pos_u)
    if (itemFeatureDim > 0)
      params.put("i", pos_i)

    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(userFeatureDim))
    if (userFeatureDim > 0)
      params.put("user_feature_dim", new Integer(userFeatureDim))
    if (itemFeatureDim > 0)
      params.put("item_feature_dim", new Integer(itemFeatureDim))
    params
  }

  def genLabelsEmbedding(batchSize: Int,
                         model: BiSAGEPSModel,
                         userFeatureDim: Int,
                         itemFeatureDim: Int,
                         numSample: Int,
                         numPartitions: Int,
                         graphType: Int,
                         taskType: String): Iterator[(Long, Long, String, String)] = {
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val batchIterator = (userTrainIdx ++ userTestIdx).sliding(batchSize, batchSize)
    val keysIterator = new Iterator[Array[(Long, Long, String, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, Long, String, String)] = {
        val batch = batchIterator.next()
        val outputs = genLabelsEmbeddingBatch(batch, model, userFeatureDim, itemFeatureDim, numSample,
          weights, torch, graphType, taskType)
        outputs.toArray
      }
    }

    keysIterator.flatMap(f => f.iterator)
  }

  def genLabelsEmbeddingBatch(batchIdx: Array[(Int, Int)],
                              model: BiSAGEPSModel,
                              userFeatureDim: Int,
                              itemFeatureDim: Int,
                              numSample: Int,
                              weights: Array[Float],
                              torch: TorchModel,
                              graphType: Int,
                              taskType: String): Iterator[(Long, Long, String, String)] = {
    val batchIds = batchIdx.map(idx => (keys(idx._1), keysI(idx._2)))
    var params = makeParams(batchIdx, numSample, userFeatureDim, itemFeatureDim, model, graphType, false)
    params.put("weights", weights)
    val embedding = torch.gcnEmbedding(params)
    assert(embedding.length % batchIds.length == 0)
    params = makeParams(embedding, batchIds.length, weights)
    val outputs = torch.gcnPred(params)
    assert(outputs.length % batchIds.length == 0)
    val numLabels = outputs.length / batchIds.length
    val outputDim = embedding.length / batchIdx.length

    if (taskType == "classification") {
      outputs.sliding(numLabels, numLabels).zip(embedding.sliding(outputDim, outputDim))
        .zip(batchIds.iterator).map {
        case ((p, h), key) =>
          val maxIndex =
            if (p.length == 1)
              if (p(0) >= 0.5) 1L
              else 0L
            else p.zipWithIndex.maxBy(_._1)._2
          (key._1, key._2, maxIndex.toString, h.mkString(","))
      }
    } else {
      outputs.sliding(numLabels, numLabels).zip(embedding.sliding(outputDim, outputDim))
        .zip(batchIds.iterator).map {
        case ((p, h), key) =>
          val maxIndex = p(0).toString
          (key._1, key._2, maxIndex, h.mkString(","))
      }
    }
  }

  def makeParams(embedding: Array[Float],
                 batchSize: Int,
                 weights: Array[Float]): JMap[String, Object] = {
    val hiddenDim = embedding.length / batchSize
    val params = new JHashMap[String, Object]()
    params.put("embedding", embedding)
    params.put("weights", weights)
    params.put("hidden_dim", new Integer(hiddenDim))
    params.put("feature_dim", new Integer(0))
    params
  }

}