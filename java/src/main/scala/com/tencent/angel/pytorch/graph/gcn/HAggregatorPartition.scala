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

import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.floats.FloatArrayList
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

import java.util.{HashMap => JHashMap, Map => JMap}

class HAggregatorPartition(index: Int,
                           keys: Array[Long],
                           indptr: Array[Int],
                           neighbors: Array[Long],
                           types: Array[Int],
                           weights: Array[Float],
                           trainIdx: Array[Int],
                           trainLabels: Array[Array[Float]],
                           testIdx: Array[Int],
                           testLabels: Array[Array[Float]],
                           torchModelPath: String,
                           useSecondOrder: Boolean) extends
    RGCNPartition(index, keys, indptr, neighbors, types, trainIdx, trainLabels,
    testIdx, testLabels, torchModelPath, useSecondOrder) {

  def genEmbedding(batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   numPartitions: Int,
                   parseAloneNodes: Boolean,
                   fieldNum: Int,
                   fieldMultiHot: Boolean,
                   cur_metapath: Integer,
                   hasUseWeightedAggregate: Boolean,
                   aggregator_in_scala: Boolean): Iterator[(Long, String)] = {

    val (new_keys, new_indptr, new_neighbors, new_types, new_weights) = filterNeighborsByMetapath(hasUseWeightedAggregate, cur_metapath)
    println("new_keys length", new_keys.length, "new_weight length", new_weights.length)
    val batchIterator = new_keys.indices.sliding(batchSize, batchSize)
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
          numSample, weights, torch, fieldNum, fieldMultiHot, hasUseWeightedAggregate, aggregator_in_scala, new_keys, new_indptr, new_neighbors, new_types, new_weights)
        output.toArray
      }
    }

    keyIterator.flatMap(f => f.iterator)
  }

  def genEmbeddingBatch(batchIdx: Array[Int],
                        model: GNNPSModel,
                        featureDim: Int,
                        numSample: Int,
                        weights: Array[Float],
                        torch: TorchModel,
                        fieldNum: Int,
                        fieldMultiHot: Boolean,
                        hasUseWeightedAggregate: Boolean,
                        aggregator_in_scala: Boolean,
                        keys: Array[Long],
                        indptr: Array[Int],
                        neighbors: Array[Long],
                        types: Array[Int],
                        edgeWeights: Array[Float]): Iterator[(Long, String)] = {

    val batchIds = batchIdx.map(idx => keys(idx))
    val (params, firstWeights) = makeParams(batchIdx, numSample, featureDim, model, false, hasUseWeightedAggregate, fieldNum, fieldMultiHot, keys, indptr, neighbors, types, edgeWeights)
    params.put("weights", weights)

    if (aggregator_in_scala) {
      println("aggregator in scala")
      val output = scatter_mean(params, featureDim, hasUseWeightedAggregate, firstWeights)
      batchIds.zip(output).iterator
    } else {
      val output = torch.gcnEmbedding(params)
      assert(output.length % batchIdx.length == 0)
      val outputDim = output.length / batchIdx.length
      println("output.length" + output.length)
      output.sliding(outputDim, outputDim)
        .zip(batchIds.iterator).map {
        case (h, key) => // h is representation for node ``key``
          (key, h.mkString(","))
      }
    }
  }

  def filterNeighborsByMetapath(hasUseWeightedAggregate: Boolean, cur_metapath : Integer): (Array[Long], Array[Int], Array[Long], Array[Int], Array[Float]) ={
    val new_keys = new LongArrayList()
    val new_indptr = new IntArrayList()
    val new_neighbors = new LongArrayList()
    val new_types = new IntArrayList()
    val new_weights = new FloatArrayList()

    new_indptr.add(0)
    var satisfy_neighbor_num = 0
    for (idx <- keys.indices) {
      val key = keys(idx)
      satisfy_neighbor_num = 0
      for (neighbor_idx <- indptr(idx) until indptr(idx+1)){
        if (types(neighbor_idx) == cur_metapath){
          satisfy_neighbor_num += 1
          new_neighbors.add(neighbors(neighbor_idx))
          new_types.add(cur_metapath)
          if (hasUseWeightedAggregate) {
            new_weights.add(weights(neighbor_idx))
          }
        }
      }
      if (satisfy_neighbor_num > 0){
        new_keys.add(key)
        new_indptr.add(new_neighbors.size())

      }
    }

    (new_keys.toLongArray(), new_indptr.toIntArray(), new_neighbors.toLongArray(), new_types.toIntArray(), new_weights.toFloatArray())
  }

  def makeParams(batchIdx: Array[Int],
                 numSample: Int,
                 featureDim: Int,
                 model: GNNPSModel,
                 isTraining: Boolean,
                 hasUseWeightedAggregate: Boolean,
                 fieldNum: Int,
                 fieldMultiHot: Boolean,
                 keys: Array[Long],
                 indptr: Array[Int],
                 neighbors: Array[Long],
                 types: Array[Int],
                 weights: Array[Float]): (JMap[String, Object], Array[Float]) = {
    val batchKeys = new LongOpenHashSet()
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val edgeTypes = new LongArrayList()
    val edgeWeights = new FloatArrayList()

    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    // Since the neighbors are filtered according to the edgetype of metapath before making parameters, it can sample like isomorphic graph
    val (first, firstTypes, firstWeights) = MakeEdgeIndex.makeEdgeIndex(batchIdx,
      keys, indptr, neighbors, weights, types, srcs, dsts, edgeTypes, edgeWeights,
      batchKeys, index, numSample, model, selfLoop = true, hasUseWeightedAggregate = hasUseWeightedAggregate)

    val params = new JHashMap[String, Object]()
    val (x, batchIds, fieldIds) = MakeSparseBiFeature.makeFeatures(index, featureDim, model, -1, params, fieldNum, fieldMultiHot)

    params.put("first_edge_index", first)

    params.put("x", x)
    if (fieldNum > 0) {
      params.put("batch_ids", batchIds)
      params.put("field_ids", fieldIds)
    }
    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(featureDim))
    (params, firstWeights)
  }


  def scatter_mean(params: JMap[String, Object], featureDim: Int, hasUseWeightedAggregate: Boolean, weights: Array[Float]): Array[String] ={
    val first = params.get("first_edge_index").asInstanceOf[Array[Long]]
    val x = params.get("x").asInstanceOf[Array[Float]].sliding(featureDim, featureDim).toArray
    val edgeNum = first.length / 2
    println("edge nums", edgeNum, "edge weight nums", weights.length)

    val src = first.take(edgeNum)
    val dst = first.drop(edgeNum)

    if (hasUseWeightedAggregate) {
      val newFeatureRDD = src.zip(dst.zip(weights))
        .groupBy(f => f._1)
        .map(r => (r._1, r._2.map(n => n._2._2).sum, r._2.map(n => x(n._2._1.toInt).map(_*n._2._2)).transpose.map(_.sum)))
        .map(r => (r._1.toInt, r._3.map(vec => vec / r._2)))
      for ((node, feature) <- newFeatureRDD){
        x(node) = feature
      }
    } else {
      val newFeatureRDD = src.zip(dst)
        .groupBy(f => f._1)
        .map(r => (r._1, r._2.length, r._2.map(n => x(n._2.toInt)).transpose.map(_.sum)))
        .map(r => (r._1.toInt, r._3.map(vec => vec / r._2)))
      for ((node, feature) <- newFeatureRDD) {
        x(node) = feature
      }
    }
    x.map(h => h.mkString(","))
  }
}
