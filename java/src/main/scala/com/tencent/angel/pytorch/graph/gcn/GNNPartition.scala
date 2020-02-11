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

import java.util.{Map => JMap}

import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel

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

  def makeParams(batchIdx: Array[Int],
                 numSample: Int,
                 featureDim: Int,
                 model: GNNPSModel,
                 isTraining: Boolean = true): JMap[String, Object] = ???

  def makeParams(nodes: Array[Long],
                 featureDim: Int,
                 model: GNNPSModel): JMap[String, Object] = ???

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int): (Double, Long, Int) = ???

  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   isTest: Boolean = true): Iterator[(Array[Float], Array[Float])] = ???

  def genLabels(batchSize: Int,
                model: GNNPSModel,
                featureDim: Int,
                numSample: Int,
                numPartitions: Int,
                parseAloneNodes: Boolean = true): Iterator[(Long, Long, String)] = ???

  //return pred and embedding
  def genLabelsEmbedding(batchSize: Int,
                         model: GNNPSModel,
                         featureDim: Int,
                         numSample: Int,
                         numPartitions: Int,
                         parseAloneNodes: Boolean = true): Iterator[(Long, Long, String, String)] = ???

  def genEmbedding(batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   numPartitions: Int,
                   parseAloneNodes: Boolean = true): Iterator[(Long, String)] = {

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
          numSample, weights, torch)
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
            featureDim, weights, torch)
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
                        torch: TorchModel): Iterator[(Long, String)] = {

    val batchIds = batchIdx.map(idx => keys(idx))
    val params = makeParams(batchIdx, numSample, featureDim, model, false)
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

  def genEmbeddingBatchAloneNodes(nodes: Array[Long],
                                  model: GNNPSModel,
                                  featureDim: Int,
                                  weights: Array[Float],
                                  torch: TorchModel): Iterator[(Long, String)] = {
    val params = makeParams(nodes, featureDim, model)
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

}
