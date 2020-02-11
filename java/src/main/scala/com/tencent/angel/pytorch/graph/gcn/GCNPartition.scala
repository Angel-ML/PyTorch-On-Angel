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

class GCNPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   trainIdx: Array[Int],
                   trainLabels: Array[Float],
                   testIdx: Array[Int],
                   testLabels: Array[Float],
                   torchModelPath: String) extends
  GNNPartition(index, keys, indptr, neighbors, torchModelPath, true) {

  def getTrainTestSize(): (Int, Int) = (trainIdx.length, testIdx.length)

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
      keys, indptr, neighbors, srcs, dsts,
      batchKeys, index, numSample, model, true)

    val x = MakeFeature.makeFeatures(index, featureDim, model)
    val params = new JHashMap[String, Object]()
    params.put("first_edge_index", first)
    params.put("second_edge_index", second)
    params.put("x", x)
    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(featureDim))
    params
  }

  override
  def makeParams(nodes: Array[Long],
                 featureDim: Int,
                 model: GNNPSModel): JMap[String, Object] = {
    val index = new Long2IntOpenHashMap()
    for (node <- nodes)
      index.put(node, index.size())
    val first = MakeEdgeIndex.makeEdgeIndex(nodes, index)
    val x = MakeFeature.makeFeatures(index, featureDim, model)
    val params = new JHashMap[String, Object]()
    params.put("first_edge_index", first)
    params.put("second_edge_index", first)
    params.put("x", x)
    params.put("batch_size", new Integer(nodes.length))
    params.put("feature_dim", new Integer(featureDim))
    params
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

  override
  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int): (Double, Long, Int) = {

    val batchIterator = trainIdx.zip(trainLabels).sliding(batchSize, batchSize)
    var lossSum = 0.0
    var numRight: Long = 0

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0

    while (batchIterator.hasNext) {
      val batch = batchIterator.next()
      val (loss, right) = trainBatch(batch, model, featureDim,
        optim, numSample, torch)
      lossSum += loss * batch.length
      numRight += right
      numStep += 1
    }

    TorchModel.put(torch) // return torch for next epoch
    (lossSum, numRight, numStep)
  }

  def trainBatch(batchIdx: Array[(Int, Float)],
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 torch: TorchModel): (Double, Long) = {

    val targets = new Array[Float](batchIdx.length)
    var k = 0
    for ((_, label) <- batchIdx) {
      targets(k) = label
      k += 1
    }

    val weights = model.readWeights()
    val params = makeParams(batchIdx.map(f => f._1), numSample, featureDim, model)
    params.put("targets", targets)
    params.put("weights", weights)
    val loss = torch.gcnBackward(params)
    model.step(weights, optim)
    (loss, 0)
  }

  override
  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   isTest: Boolean = true): Iterator[(Array[Float], Array[Float])] = {

    val zips = if (isTest) testIdx.zip(testLabels) else trainIdx.zip(trainLabels)
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
        predictBatch(batch, model, featureDim, numSample, weights, torch)
      }
    }
  }

  def predictBatch(batchIdx: Array[(Int, Float)],
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   weights: Array[Float],
                   torch: TorchModel): (Array[Float], Array[Float]) = {
    val targets = batchIdx.map(f => f._2)
    val params = makeParams(batchIdx.map(f => f._1), numSample, featureDim, model, false)
    params.put("weights", weights)
    val outputs = torch.gcnPredict(params)
    outputs match {
      case f: Array[Float] => (targets, f)
      case l: Array[Long] => (targets, l.map(x => x.toFloat))
    }
  }

  override
  def genLabels(batchSize: Int,
                model: GNNPSModel,
                featureDim: Int,
                numSample: Int,
                numPartitions: Int,
                parseAloneNodes: Boolean = true): Iterator[(Long, Long, String)] = {

    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    val keysIterator = new Iterator[Array[(Long, Long, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, Long, String)] = {
        val batch = batchIterator.next().toArray
        val outputs = genLabelsBatch(batch, model, featureDim, numSample,
          weights, torch)
        outputs.toArray
      }
    }

    if (parseAloneNodes) { // whether generating labels for nodes without edges
      val aloneNodes = model.getNodesWithOutDegree(index, numPartitions)
      val aloneBatchIterator = aloneNodes.sliding(batchSize, batchSize)
      val aloneIterator = new Iterator[Array[(Long, Long, String)]] with Serializable {
        override def hasNext: Boolean = aloneBatchIterator.hasNext

        override def next: Array[(Long, Long, String)] = {
          val batch = aloneBatchIterator.next()
          val outputs = genLabelsBatchAloneNodes(batch, model,
            featureDim, weights, torch)
          outputs.toArray
        }
      }

      aloneIterator.flatMap(f => f.iterator) ++ keysIterator.flatMap(f => f.iterator)
    } else {
      keysIterator.flatMap(f => f.iterator)
    }
  }

  def genLabelsBatch(batchIdx: Array[Int],
                     model: GNNPSModel,
                     featureDim: Int,
                     numSample: Int,
                     weights: Array[Float],
                     torch: TorchModel): Iterator[(Long, Long, String)] = {


    val batchIds = batchIdx.map(idx => keys(idx))
    val params = makeParams(batchIdx, numSample, featureDim, model, false)
    params.put("weights", weights)
    val outputs = torch.gcnForward(params)
    assert(outputs.length % batchIds.length == 0)
    val numLabels = outputs.length / batchIds.length
    outputs.sliding(numLabels, numLabels)
      .zip(batchIds.iterator).map {
      case (p, key) =>
        val maxIndex =
          if (p.length == 1)
            if (p(0) >= 0.5) 1L
            else 0L
          else p.zipWithIndex.maxBy(_._1)._2
        (key, maxIndex, p.mkString(","))
    }
  }

  def genLabelsBatchAloneNodes(nodes: Array[Long],
                               model: GNNPSModel,
                               featureDim: Int,
                               weights: Array[Float],
                               torch: TorchModel): Iterator[(Long, Long, String)] = {
    val params = makeParams(nodes, featureDim, model)
    params.put("weights", weights)
    val outputs = torch.gcnForward(params)
    assert(outputs.length % nodes.length == 0)
    val numLabels = outputs.length / nodes.length
    outputs.sliding(numLabels, numLabels)
      .zip(nodes.iterator).map {
      case (p, key) =>
        val maxIndex =
          if (p.length == 1)
            if (p(0) >= 0.5) 1L
            else 0L
          else p.zipWithIndex.maxBy(_._1)._2
        (key, maxIndex, p.mkString(","))
    }
  }


  override def genLabelsEmbedding(batchSize: Int,
                                  model: GNNPSModel,
                                  featureDim: Int,
                                  numSample: Int,
                                  numPartitions: Int,
                                  parseAloneNodes: Boolean = true): Iterator[(Long, Long, String, String)] = {
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    val keysIterator = new Iterator[Array[(Long, Long, String, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, Long, String, String)] = {
        val batch = batchIterator.next().toArray
        val outputs = genLabelsEmbeddingBatch(batch, model, featureDim, numSample,
          weights, torch)
        outputs.toArray
      }
    }

    if (parseAloneNodes) { // whether generating labels for nodes without edges
      val aloneNodes = model.getNodesWithOutDegree(index, numPartitions)
      val aloneBatchIterator = aloneNodes.sliding(batchSize, batchSize)
      val aloneIterator = new Iterator[Array[(Long, Long, String, String)]] with Serializable {
        override def hasNext: Boolean = aloneBatchIterator.hasNext

        override def next: Array[(Long, Long, String, String)] = {
          val batch = aloneBatchIterator.next()
          val outputs = genLabelsEmbeddingBatchAloneNodes(batch, model,
            featureDim, weights, torch)
          outputs.toArray
        }
      }

      aloneIterator.flatMap(f => f.iterator) ++ keysIterator.flatMap(f => f.iterator)
    } else {
      keysIterator.flatMap(f => f.iterator)
    }
  }

  def genLabelsEmbeddingBatch(batchIdx: Array[Int],
                              model: GNNPSModel,
                              featureDim: Int,
                              numSample: Int,
                              weights: Array[Float],
                              torch: TorchModel): Iterator[(Long, Long, String, String)] = {
    val batchIds = batchIdx.map(idx => keys(idx))
    var params = makeParams(batchIdx, numSample, featureDim, model, false)
    params.put("weights", weights)
    val embedding = torch.gcnEmbedding(params)
    assert(embedding.length % batchIds.length == 0)
    params = makeParams(embedding, batchIds.length, weights)
    val outputs = torch.gcnPred(params)
    assert(outputs.length % batchIds.length == 0)
    val numLabels = outputs.length / batchIds.length
    val outputDim = embedding.length / batchIdx.length
    outputs.sliding(numLabels, numLabels).zip(embedding.sliding(outputDim, outputDim))
      .zip(batchIds.iterator).map {
      case ((p, h), key) =>
        val maxIndex =
          if (p.length == 1)
            if (p(0) >= 0.5) 1L
            else 0L
          else p.zipWithIndex.maxBy(_._1)._2
        (key, maxIndex, h.mkString(","), p.mkString(","))
    }
  }

  def genLabelsEmbeddingBatchAloneNodes(nodes: Array[Long],
                                        model: GNNPSModel,
                                        featureDim: Int,
                                        weights: Array[Float],
                                        torch: TorchModel): Iterator[(Long, Long, String, String)] = {
    var params = makeParams(nodes, featureDim, model)
    params.put("weights", weights)
    val embedding = torch.gcnEmbedding(params)
    assert(embedding.length % nodes.length == 0)
    params = makeParams(embedding, nodes.length, weights)
    val outputs = torch.gcnPred(params)
    assert(outputs.length % nodes.length == 0)
    val numLabels = outputs.length / nodes.length
    val outputDim = embedding.length / nodes.length
    outputs.sliding(numLabels, numLabels).zip(embedding.sliding(outputDim, outputDim))
      .zip(nodes.iterator).map {
      case ((p, h), key) =>
        val maxIndex =
          if (p.length == 1)
            if (p(0) >= 0.5) 1L
            else 0L
          else p.zipWithIndex.maxBy(_._1)._2
        (key, maxIndex, h.mkString(","), p.mkString(","))
    }
  }

}
