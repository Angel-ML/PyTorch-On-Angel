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

import java.util.{Random, Map => JMap}

import com.tencent.angel.ml.math2.storage.IntFloatSparseVectorStorage
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.ints.Int2FloatOpenHashMap

class GNNPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   torchModelPath: String,
                   useSecondOrder: Boolean) extends Serializable {

  def numNodes: Long = keys.length

  def numEdges: Long = neighbors.length

  def aloneNodes(model: GNNPSModel, numPartitions: Int): Array[Long] =
    model.getNodesWithOutDegree(index, numPartitions)

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): (Double, Long, Int) = ???

  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   isTest: Boolean,
                   fieldNum: Int,
                   fieldMultiHot: Boolean): Iterator[(Array[Float], Array[Float])] = ???

  def genLabels(batchSize: Int,
                model: GNNPSModel,
                featureDim: Int,
                numSample: Int,
                numPartitions: Int,
                multiLabelsNum: Int,
                parseAloneNodes: Boolean,
                fieldNum: Int,
                fieldMultiHot: Boolean): Iterator[(Long, String, String)] = ???

  //return pred and embedding
  def genLabelsEmbedding(batchSize: Int,
                         model: GNNPSModel,
                         featureDim: Int,
                         numSample: Int,
                         numPartitions: Int,
                         multiLabelsNum: Int,
                         parseAloneNodes: Boolean,
                         fieldNum: Int,
                         fieldMultiHot: Boolean): Iterator[(Long, String, String, String)] = ???

  def genEmbedding(batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   numPartitions: Int,
                   parseAloneNodes: Boolean,
                   fieldNum: Int,
                   fieldMultiHot: Boolean): Iterator[(Long, String)] = {

    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()

    val keyIterator = new Iterator[Array[(Long, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, String)] = {
        val batch = batchIterator.next().toArray
        val output = genEmbeddingBatch(batch, model, featureDim,
          numSample, weights, torch, fieldNum, fieldMultiHot)
        output.toArray
      }
    }

    if (parseAloneNodes) { // whether parsing nodes without edges
      val aloneNodes = model.getNodesWithOutDegree(index, numPartitions)
      val aloneBatchIterator = aloneNodes.sliding(batchSize, batchSize)
      val aloneIterator = new Iterator[Array[(Long, String)]] with Serializable {
        override def hasNext: Boolean = aloneBatchIterator.hasNext


        override def next: Array[(Long, String)] = {
          val batch = aloneBatchIterator.next()
          val outputs = genEmbeddingBatchAloneNodes(batch, model,
            featureDim, weights, torch, fieldNum, fieldMultiHot)
          outputs.toArray
        }
      }

      aloneIterator.flatMap(f => f.iterator) ++ keyIterator.flatMap(f => f.iterator)
    } else {
      keyIterator.flatMap(f => f.iterator)
    }
  }

  def genEmbeddingBatch(batchIdx: Array[Int],
                        model: GNNPSModel,
                        featureDim: Int,
                        numSample: Int,
                        weights: Array[Float],
                        torch: TorchModel,
                        fieldNum: Int,
                        fieldMultiHot: Boolean): Iterator[(Long, String)] = {

    val batchIds = batchIdx.map(idx => keys(idx))
    val params = makeParams(batchIdx, numSample, featureDim, model, false, fieldNum, fieldMultiHot)
    params.put("weights", weights)
    val output = torch.gcnEmbedding(params)
    assert(output.length % batchIdx.length == 0)
    val outputDim = output.length / batchIdx.length
    output.sliding(outputDim, outputDim)
      .zip(batchIds.iterator).map {
      case (h, key) => // h is representation for node ``key``
        (key, h.mkString(","))
    }
  }

  def makeParams(batchIdx: Array[Int],
                 numSample: Int,
                 featureDim: Int,
                 model: GNNPSModel,
                 isTraining: Boolean = true,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): JMap[String, Object] = ???

  def genEmbeddingBatchAloneNodes(nodes: Array[Long],
                                  model: GNNPSModel,
                                  featureDim: Int,
                                  weights: Array[Float],
                                  torch: TorchModel,
                                  fieldNum: Int,
                                  fieldMultiHot: Boolean): Iterator[(Long, String)] = {
    val params = makeParams(nodes, featureDim, model, fieldNum, fieldMultiHot)
    params.put("weights", weights)
    val output = torch.gcnEmbedding(params)
    assert(output.length % nodes.length == 0)
    val outputDim = output.length / nodes.length
    output.sliding(outputDim, outputDim)
      .zip(nodes.iterator).map {
      case (h, key) => // h is representation for node ``key``
        (key, h.mkString(","))
    }
  }

  def makeParams(nodes: Array[Long],
                 featureDim: Int,
                 model: GNNPSModel,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): JMap[String, Object] = ???

  def makeEmbeddingGrad(grad: Array[Float], embedding: Array[Vector], featIds: Array[Int], embeddingDim: Int): Unit = {
    val grads = embedding.map(f => new Int2FloatOpenHashMap(f.getSize.toInt))

    for (i <- featIds.indices) {
      for (j <- 0 until embeddingDim) {
        grads(j).addTo(featIds(i), grad(i * embeddingDim + j))
      }
    }

    embedding.zip(grads).foreach {
      case (e, g) => e.setStorage(new IntFloatSparseVectorStorage(e.dim().toInt, g))
    }
  }

  def sampleTrainData(indices: Range, ratio: Float): IndexedSeq[Int] = {
    if (ratio == 1.0f) {
      indices
    } else {
      val random = new Random(System.currentTimeMillis())
      val len = indices.length
      val start = (random.nextDouble() * len).toInt
      val count = (ratio * len).toInt
      if ((start + count) < len) {
        indices.slice(start, start + count)
      } else {
        indices.slice(start, len).++(indices.slice(0, count - (len - start)))
      }
    }
  }

}
