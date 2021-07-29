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
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

class GCNPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   trainIdx: Array[Int],
                   trainLabels: Array[Array[Float]],
                   testIdx: Array[Int],
                   testLabels: Array[Array[Float]],
                   torchModelPath: String,
                   useSecondOrder: Boolean) extends
  GNNPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder) {

  var params: JMap[String, Object] = _

  def getTrainTestSize(): (Int, Int) = (trainIdx.length, testIdx.length)

  override
  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): (Double, Long, Int) = {

    val batchIterator = trainIdx.zip(trainLabels).sliding(batchSize, batchSize)
    var lossSum = 0.0
    var numRight: Long = 0

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0

    val pairs = new Iterator[Array[(Float, Float)]] with Serializable {
      override def hasNext: Boolean = {
        batchIterator.hasNext
      }

      override def next: Array[(Float, Float)] = {
        val batch = batchIterator.next()
        val (loss, output) = trainBatch(batch, model, featureDim, optim, numSample, torch, false, fieldNum, fieldMultiHot)
        lossSum += loss * batch.length
        numRight += 0
        numStep += 1
        output._1.zip(output._2)
      }
    }.flatMap(p =>p).toArray

    TorchModel.put(torch) // return torch for next epoch
    (lossSum, numRight, numStep)
  }

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 isSharedSamples: Boolean,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): (Double, Long, Int, Array[(Float, Float)]) = {

    val batchIterator = trainIdx.zip(trainLabels).sliding(batchSize, batchSize)
    var lossSum = 0.0
    var numRight: Long = 0

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0

    val pairs = new Iterator[Array[(Float, Float)]] with Serializable {
      override def hasNext: Boolean = {
        batchIterator.hasNext
      }

      override def next: Array[(Float, Float)] = {
        val batch = batchIterator.next()
        val (loss, output) = trainBatch(batch, model, featureDim, optim, numSample, torch, isSharedSamples, fieldNum, fieldMultiHot)
        lossSum += loss * batch.length
        numRight += 0
        numStep += 1
        output._1.zip(output._2)
      }
    }.flatMap(p =>p).toArray

    TorchModel.put(torch) // return torch for next epoch
    (lossSum, numRight, numStep, pairs)
  }

  def trainBatch(batchIdx: Array[(Int, Array[Float])],
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 torch: TorchModel,
                 isSharedSamples: Boolean,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): (Double, (Array[Float], Array[Float])) = {

    val targets = batchIdx.flatMap(_._2)

    var weights = model.readWeights()
    val params = makeParams(batchIdx.map(f => f._1), numSample, featureDim, model, true, fieldNum, fieldMultiHot)
    params.put("targets", targets)
    params.put("weights", weights)

    val loss = torch.gcnBackward(params, fieldNum > 0)
    model.step(weights, optim)

    // update grad for embedding of sparse
    if (fieldNum > 0) {
      //check if the grad really replaced the pulledUEmbedding
      val pulledEmbedding = params.get("pulledEmbedding").asInstanceOf[Array[Vector]]
      val feats = params.get("feats").asInstanceOf[Array[Int]]
      val embeddingInput = params.get("x").asInstanceOf[Array[Float]]
      val embeddingDim = pulledEmbedding.length
      makeEmbeddingGrad(embeddingInput, pulledEmbedding, feats, embeddingDim)
      model.asInstanceOf[SparseGNNPSModel].updateEmbedding(pulledEmbedding, optim)
    }

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

  override
  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   isTest: Boolean,
                   fieldNum: Int,
                   fieldMultiHot: Boolean): Iterator[(Array[Float], Array[Float])] = {

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
        predictBatch(batch, model, featureDim, numSample, weights, torch, fieldNum, fieldMultiHot)
      }
    }
  }

  def predictBatch(batchIdx: Array[(Int, Array[Float])],
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   weights: Array[Float],
                   torch: TorchModel,
                   fieldNum: Int,
                   fieldMultiHot: Boolean): (Array[Float], Array[Float]) = {
    val targets = batchIdx.flatMap(f => f._2)
    val params = makeParams(batchIdx.map(f => f._1), numSample, featureDim, model, false, fieldNum, fieldMultiHot)
    params.put("weights", weights)
    val outputs = torch.gcnPredict(params)
    outputs match {
      case f: Array[Float] => (targets, f)
      case l: Array[Long] => (targets, l.map(x => x.toFloat))
    }
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

  override
  def genLabels(batchSize: Int,
                model: GNNPSModel,
                featureDim: Int,
                numSample: Int,
                numPartitions: Int,
                multiLabelsNum: Int,
                parseAloneNodes: Boolean = true,
                fieldNum: Int,
                fieldMultiHot: Boolean): Iterator[(Long, String, String)] = {

    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    val keysIterator = new Iterator[Array[(Long, String, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, String, String)] = {
        val batch = batchIterator.next().toArray
        val outputs = genLabelsBatch(batch, model, featureDim, numSample,
          weights, torch, multiLabelsNum, fieldNum, fieldMultiHot)
        outputs.toArray
      }
    }

    if (parseAloneNodes) { // whether generating labels for nodes without edges
      val aloneNodes = model.getNodesWithOutDegree(index, numPartitions)
      val aloneBatchIterator = aloneNodes.sliding(batchSize, batchSize)
      val aloneIterator = new Iterator[Array[(Long, String, String)]] with Serializable {
        override def hasNext: Boolean = aloneBatchIterator.hasNext

        override def next: Array[(Long, String, String)] = {
          val batch = aloneBatchIterator.next()
          val outputs = genLabelsBatchAloneNodes(batch, model,
            featureDim, weights, torch, multiLabelsNum, fieldNum, fieldMultiHot)
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
                     torch: TorchModel,
                     multiLabelsNum: Int,
                     fieldNum: Int,
                     fieldMultiHot: Boolean): Iterator[(Long, String, String)] = {

    val batchIds = batchIdx.map(idx => keys(idx))
    val params = makeParams(batchIdx, numSample, featureDim, model, false, fieldNum, fieldMultiHot)
    params.put("weights", weights)
    val outputs = torch.gcnForward(params)
    assert(outputs.length % batchIds.length == 0)
    val numLabels = outputs.length / batchIds.length
    outputs.sliding(numLabels, numLabels)
      .zip(batchIds.iterator).map {
      case (p, key) =>
        val maxIndex =
          if (multiLabelsNum == 1) {
            if (p.length == 1)
              if (p(0) >= 0.5) 1L.toString else 0L.toString
            else p.zipWithIndex.maxBy(_._1)._2.toString
          } else {
            p.map(x => if (x >= 0.5) 1 else 0).mkString(",")
          }
        (key, maxIndex, p.mkString(","))
    }
  }

  def genLabelsBatchAloneNodes(nodes: Array[Long],
                               model: GNNPSModel,
                               featureDim: Int,
                               weights: Array[Float],
                               torch: TorchModel,
                               multiLabelsNum: Int,
                               fieldNum: Int,
                               fieldMultiHot: Boolean): Iterator[(Long, String, String)] = {
    val params = makeParams(nodes, featureDim, model, fieldNum, fieldMultiHot)
    params.put("weights", weights)
    val outputs = torch.gcnForward(params)
    assert(outputs.length % nodes.length == 0)
    val numLabels = outputs.length / nodes.length
    outputs.sliding(numLabels, numLabels)
      .zip(nodes.iterator).map {
      case (p, key) =>
        val maxIndex =if (multiLabelsNum == 1) {
          if (p.length == 1)
            if (p(0) >= 0.5) 1L.toString else 0L.toString
          else p.zipWithIndex.maxBy(_._1)._2.toString
        } else {
          p.map(x => if (x >= 0.5) 1 else 0).mkString(",")
        }
        (key, maxIndex, p.mkString(","))
    }
  }

  override def genLabelsEmbedding(batchSize: Int,
                                  model: GNNPSModel,
                                  featureDim: Int,
                                  numSample: Int,
                                  numPartitions: Int,
                                  multiLabelsNum: Int,
                                  parseAloneNodes: Boolean = true,
                                  fieldNum: Int,
                                  fieldMultiHot: Boolean): Iterator[(Long, String, String, String)] = {
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    val keysIterator = new Iterator[Array[(Long, String, String, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, String, String, String)] = {
        val batch = batchIterator.next().toArray
        val outputs = genLabelsEmbeddingBatch(batch, model, featureDim, numSample,
          weights, torch, multiLabelsNum, fieldNum, fieldMultiHot)
        outputs.toArray
      }
    }

    if (parseAloneNodes) { // whether generating labels for nodes without edges
      val aloneNodes = model.getNodesWithOutDegree(index, numPartitions)
      val aloneBatchIterator = aloneNodes.sliding(batchSize, batchSize)
      val aloneIterator = new Iterator[Array[(Long, String, String, String)]] with Serializable {
        override def hasNext: Boolean = aloneBatchIterator.hasNext

        override def next: Array[(Long, String, String, String)] = {
          val batch = aloneBatchIterator.next()
          val outputs = genLabelsEmbeddingBatchAloneNodes(batch, model,
            featureDim, weights, torch, multiLabelsNum, fieldNum, fieldMultiHot)
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
                              torch: TorchModel,
                              multiLabelsNum: Int,
                              fieldNum: Int,
                              fieldMultiHot: Boolean): Iterator[(Long, String, String, String)] = {
    val batchIds = batchIdx.map(idx => keys(idx))
    var params = makeParams(batchIdx, numSample, featureDim, model, false, fieldNum, fieldMultiHot)
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
          if (multiLabelsNum == 1) {
            if (p.length == 1)
              if (p(0) >= 0.5) 1L.toString else 0L.toString
            else p.zipWithIndex.maxBy(_._1)._2.toString
          } else {
            p.map(x => if (x >= 0.5) 1 else 0).mkString(",")
          }
        (key, maxIndex, h.mkString(","), p.mkString(","))
    }
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

    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx,
      keys, indptr, neighbors, srcs, dsts,
      batchKeys, index, numSample, model, useSecondOrder)

    val params = new JHashMap[String, Object]()
    val (x, batchIds, fieldIds)  = MakeSparseBiFeature.makeFeatures(index, featureDim, model, -1, params, fieldNum, fieldMultiHot)

    params.put("first_edge_index", first)
    if (useSecondOrder) params.put("second_edge_index", second)
    params.put("x", x)
    if (fieldNum > 0) {
      params.put("batch_ids", batchIds)
      params.put("field_ids", fieldIds)
    }
    params.put("batch_size", new Integer(batchIdx.length))
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

  def genLabelsEmbeddingBatchAloneNodes(nodes: Array[Long],
                                        model: GNNPSModel,
                                        featureDim: Int,
                                        weights: Array[Float],
                                        torch: TorchModel,
                                        multiLabelsNum: Int,
                                        fieldNum: Int,
                                        fieldMultiHot: Boolean): Iterator[(Long, String, String, String)] = {
    var params = makeParams(nodes, featureDim, model, fieldNum, fieldMultiHot)
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
          if (multiLabelsNum == 1) {
            if (p.length == 1)
              if (p(0) >= 0.5) 1L.toString else 0L.toString
            else p.zipWithIndex.maxBy(_._1)._2.toString
          } else {
            p.map(x => if (x >= 0.5) 1 else 0).mkString(",")
          }
        (key, maxIndex, h.mkString(","), p.mkString(","))
    }
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
    val params = new JHashMap[String, Object]()
    val (x, batchIds, fieldIds) = MakeSparseBiFeature.makeFeatures(index, featureDim, model, -1, params, fieldNum,
      fieldMultiHot)
    params.put("first_edge_index", first)
    if (useSecondOrder) params.put("second_edge_index", first)
    params.put("x", x)
    if (fieldNum > 0) {
      params.put("batch_ids", batchIds)
      params.put("field_ids", fieldIds)
    }
    params.put("batch_size", new Integer(nodes.length))
    params.put("feature_dim", new Integer(featureDim))
    params
  }

}
