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
import java.util.Random

import com.tencent.angel.graph.client.psf.sample.sampleneighbor.SampleType
import com.tencent.angel.ml.math2.storage.IntFloatDenseVectorStorage
import com.tencent.angel.ml.math2.vector.IntFloatVector

import scala.collection.JavaConversions._
import it.unimi.dsi.fastutil.floats.FloatArrayList
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, Long2ObjectOpenHashMap, LongArrayList, LongArrays, LongOpenHashSet}

import scala.collection.mutable.ArrayBuffer

object MakeEdgeIndex {

  /*
    Make edge Index for algorithms without edge types, including GCN, GraphSage and DGI.
   */
  def makeEdgeIndex(batchIdx: Array[Int],
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    batchKeys: LongOpenHashSet,
                    index: Long2IntOpenHashMap,
                    numSample: Int,
                    model: GNNPSModel,
                    useSecondOrder: Boolean): (Array[Long], Array[Long]) = {
    if (!useSecondOrder) {
      val edgeIndex = makeEdgeIndex(batchIdx, keys, indptr, neighbors,
        srcs, dsts, batchKeys, index, numSample)
      (edgeIndex, null)
    } else {
      val (first, firstKeys) = makeFirstOrderEdgeIndex(
        batchIdx, keys, indptr, neighbors,
        srcs, dsts, batchKeys, index, numSample)

      if (firstKeys.size() == 0) // for nodes without out-edges
        return (first, first)

      val second = makeSecondOrderEdgeIndex(batchKeys, firstKeys,
        srcs, dsts, index, model, numSample)
      (first, second)
    }
  }

  /*
    Make edge Index for algorithms with edge weights, including DGI.
   */
  def makeEdgeIndex(batchIdx: Array[Int],
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    weights: Array[Float],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    edgeWights: FloatArrayList,
                    batchKeys: LongOpenHashSet,
                    index: Long2IntOpenHashMap,
                    numSample: Int,
                    model: GNNPSModel,
                    useSecondOrder: Boolean): (Array[Long], Array[Long]) = {
    if (!useSecondOrder) {
      val edgeIndex = makeEdgeIndex(batchIdx, keys, indptr, neighbors, weights,
        srcs, dsts, edgeWights, batchKeys, index, numSample)
      (edgeIndex, null)
    } else {
      val (first, firstKeys) = makeFirstOrderEdgeIndex(
        batchIdx, keys, indptr, neighbors,
        srcs, dsts, batchKeys, index, numSample)

      if (firstKeys.size() == 0) // for nodes without out-edges
        return (first, first)

      val second = makeSecondOrderEdgeIndex(batchKeys, firstKeys,
        srcs, dsts, index, model, numSample)
      (first, second)
    }
  }

  /*
    Make edge Index for algorithms without edge types, including GAT.
   */
  def makeEdgeIndexForAllEdge(batchIdx: Array[Int],
                              keys: Array[Long],
                              indptr: Array[Int],
                              neighbors: Array[Long],
                              srcs: LongArrayList,
                              dsts: LongArrayList,
                              batchKeys: LongOpenHashSet,
                              index: Long2IntOpenHashMap,
                              numSample: Int,
                              model: GNNPSModel,
                              useSecondOrder: Boolean): (Array[Long], Array[Long]) = {
    if (!useSecondOrder) {
      val edgeIndex = makeEdgeIndex(batchIdx, keys, indptr, neighbors,
        srcs, dsts, batchKeys, index, numSample)
      (edgeIndex, null)
    } else {
      val (first, firstKeys) = makeFirstOrderEdgeIndexForAllEdge(
        batchIdx, keys, indptr, neighbors,
        srcs, dsts, batchKeys, index, numSample)

      if (firstKeys.size() == 0) // for nodes without out-edges
        return (first, first)

      val second = makeSecondOrderEdgeIndex(batchKeys, firstKeys,
        srcs, dsts, index, model, numSample)
      (first, second)
    }
  }



  /*
    Make edge Index for only one-order algorithm, for example DGI with one-order.
    This function do not return the firstKeys, which contains the sources for second-order
    neighbor graph.
   */
  private
  def makeEdgeIndex(batchIdx: Array[Int],
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    batchKeys: LongOpenHashSet,
                    index: Long2IntOpenHashMap,
                    numSample: Int): Array[Long] = {
    val random = new Random(System.currentTimeMillis())
    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = index.get(key)
      srcs.add(keyIndex) // add self-loop
      dsts.add(keyIndex)
      val len = indptr(idx + 1) - indptr(idx)
      if (len > numSample)
        LongArrays.shuffle(neighbors, indptr(idx), indptr(idx + 1), random)
      val size = math.min(len, numSample)
      var j = indptr(idx)
      while (j < indptr(idx) + size) {
        val n = neighbors(j)
        if (!index.containsKey(n))
          index.put(n, index.size())
        srcs.add(keyIndex)
        dsts.add(index.get(n))
        j += 1
      }
    }
    makeEdgeIndexFromSrcDst(srcs, dsts)
  }

  /*
  for one-order algorithm with weighted, such as DGI
   */
  def makeEdgeIndex(batchIdx: Array[Int],
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    weights: Array[Float],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    edgeWeights: FloatArrayList,
                    batchKeys: LongOpenHashSet,
                    index: Long2IntOpenHashMap,
                    numSample: Int): Array[Long] = {
    val random = new Random(System.currentTimeMillis())
    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = index.get(key)
      srcs.add(keyIndex) // add self-loop
      dsts.add(keyIndex)
      edgeWeights.add(0.5f)
      val len = indptr(idx + 1) - indptr(idx)
      var start = 0
      if (len > numSample)
        start = random.nextInt(len)
      val size = math.min(len, numSample)
      for (j <- 0 until size) {
        val jj = (j + start) % len + indptr(idx)
        val n = neighbors(jj)
        if (!index.containsKey(n))
          index.put(n, index.size())
        srcs.add(keyIndex)
        dsts.add(index.get(n))
        edgeWeights.add(weights(j))
      }
    }
    makeEdgeIndexFromSrcDst(srcs, dsts)
  }

  /*
    Make edge Index for algorithms with edge features, including EdgePropGCN
   */
  def makeEdgeIndex(batchIdx: Array[Int],
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    edgeDim: Int,
                    features: Array[IntFloatVector],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    edgeFeatures: util.ArrayList[IntFloatVector],
                    batchKeys: LongOpenHashSet,
                    index: Long2IntOpenHashMap,
                    numSample: Int,
                    model: GNNPSModel,
                    useSecondOrder: Boolean): (Array[Long], Array[Long], Array[Float], Array[Float]) = {
    if (!useSecondOrder) {//todo???
    val ((edgeIndex, edgeFeature), _) = makeFirstOrderEdgeIndex(batchIdx, keys, indptr, neighbors, edgeDim, features,
      srcs, dsts, edgeFeatures, batchKeys, index, numSample)
      (edgeIndex, null, edgeFeature, null)
    } else {
      val ((first, firstFeature), firstKeys) = makeFirstOrderEdgeIndex(
        batchIdx, keys, indptr, neighbors, edgeDim, features,
        srcs, dsts, edgeFeatures, batchKeys, index, numSample)

      if (firstKeys.size() == 0) // for nodes without out-edges
        return (first, first, firstFeature, firstFeature)

      val (second, secondFeature) = makeSecondOrderEdgeIndex(batchKeys, firstKeys,
        srcs, dsts, edgeFeatures, index, model, numSample, edgeDim)
      (first, second, firstFeature, secondFeature)
    }
  }

  /*
    Make edge Index for algorithms with edge features abd edge types
   */
  def makeEdgeIndex(batchIdx: Array[Int],
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    edgeDim: Int,
                    features: Array[IntFloatVector],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    edgeTypes: LongArrayList,
                    edgeFeatures: util.ArrayList[IntFloatVector],
                    batchKeys: LongOpenHashSet,
                    index: Long2IntOpenHashMap,
                    numSample: Int,
                    model: GNNPSModel,
                    useSecondOrder: Boolean): (Array[Long], Array[Long], Array[Float], Array[Float], Array[Long], Array[Long]) = {
    if (!useSecondOrder) {//todo???
    val ((edgeIndex, edgeFeature), edgeType, _) = makeFirstOrderEdgeIndex(
      batchIdx, keys, indptr, neighbors, types, edgeDim, features,
      srcs, dsts, edgeTypes, edgeFeatures, batchKeys, index, numSample)
      (edgeIndex, null, edgeFeature, null, edgeType, null)
    } else {
      val ((first, firstFeature), firstEdgeType, firstKeys) = makeFirstOrderEdgeIndex(
        batchIdx, keys, indptr, neighbors, types, edgeDim, features,
        srcs, dsts, edgeTypes, edgeFeatures, batchKeys, index, numSample)

      if (firstKeys.size() == 0) // for nodes without out-edges
        return (first, first, firstFeature, firstFeature, firstEdgeType, firstEdgeType)

      val ((second, secondFeature), secondEdgeType) = makeSecondOrderEdgeIndex(batchKeys, firstKeys,
        srcs, dsts, edgeTypes,edgeFeatures, index, model, numSample, edgeDim, edgeTypes == null)
      (first, second, firstFeature, secondFeature, firstEdgeType, secondEdgeType)
    }
  }

  /*
    Make edge Index for algorithms requires two-order neighbor graph.
    We return the firstKeys as a hashset, which contains the sources nodes for the
    two-order neighbor graph.
   */
  private
  def makeFirstOrderEdgeIndex(batchIdx: Array[Int],
                              keys: Array[Long],
                              indptr: Array[Int],
                              neighbors: Array[Long],
                              srcs: LongArrayList,
                              dsts: LongArrayList,
                              batchKeys: LongOpenHashSet,
                              index: Long2IntOpenHashMap,
                              numSample: Int): (Array[Long], LongOpenHashSet) = {
    val random = new Random(System.currentTimeMillis())
    val firstKeys = new LongOpenHashSet()
    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = index.get(key)
      srcs.add(keyIndex) // add self-loop
      dsts.add(keyIndex)
      val len = indptr(idx + 1) - indptr(idx)
      if (len > numSample)
        LongArrays.shuffle(neighbors, indptr(idx), indptr(idx + 1), random)
      val size = math.min(len, numSample)
      var j = indptr(idx)
      while (j < indptr(idx) + size) {
        val n = neighbors(j)
        if (!batchKeys.contains(n))
          firstKeys.add(n)
        if (!index.containsKey(n))
          index.put(n, index.size())
        srcs.add(keyIndex)
        dsts.add(index.get(n))
        j += 1
      }
    }

    (makeEdgeIndexFromSrcDst(srcs, dsts), firstKeys)
  }

  private
  def makeFirstOrderEdgeIndexForAllEdge(batchIdx: Array[Int],
                                        keys: Array[Long],
                                        indptr: Array[Int],
                                        neighbors: Array[Long],
                                        srcs: LongArrayList,
                                        dsts: LongArrayList,
                                        batchKeys: LongOpenHashSet,
                                        index: Long2IntOpenHashMap,
                                        numSample: Int): (Array[Long], LongOpenHashSet) = {
    val random = new Random(System.currentTimeMillis())
    val firstKeys = new LongOpenHashSet()
    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = index.get(key)
      srcs.add(keyIndex) // add self-loop
      dsts.add(keyIndex)
      val len = indptr(idx + 1) - indptr(idx)
      if (len > numSample)
        LongArrays.shuffle(neighbors, indptr(idx), indptr(idx + 1), random)
      val size = if (numSample < 0) len else math.min(len, numSample)
      var j = indptr(idx)
      while (j < indptr(idx) + size) {
        val n = neighbors(j)
        if (!batchKeys.contains(n))
          firstKeys.add(n)
        if (!index.containsKey(n))
          index.put(n, index.size())
        srcs.add(keyIndex)
        dsts.add(index.get(n))
        j += 1
      }
    }

    (makeEdgeIndexFromSrcDst(srcs, dsts), firstKeys)
  }

  private
  def makeSecondOrderEdgeIndex(batchKeys: LongOpenHashSet,
                               firstKeys: LongOpenHashSet,
                               srcs: LongArrayList,
                               dsts: LongArrayList,
                               index: Long2IntOpenHashMap,
                               model: GNNPSModel,
                               numSample: Int): Array[Long] = {
    val (edgeIndex, _) = makeSecondOrderEdgeIndex(batchKeys, firstKeys, srcs, dsts,
      null, index, model, numSample, true)
    edgeIndex
  }

  /*
  Make edge index for HAN for input graphs with meta-paths as "user-item-user"
  1. delete first srcs and dsts indexes from the second ones, since HAN ignore the intermediate item nodes,
     return the second indexes and types
  2. add self loop afterward, assign the self loop edge type as itemTypes param
 */
  def makeHANEdgeIndex(batchIdx: Array[Int],
                       keys: Array[Long],
                       indptr: Array[Int],
                       neighbors: Array[Long],
                       types: Array[Int],
                       srcs: LongArrayList,
                       dsts: LongArrayList,
                       edgeTypes: LongArrayList,
                       index: Long2IntOpenHashMap,
                       numSample: Int,
                       model: GNNPSModel,
                       itemTypes: Int): (Array[Long], Array[Long]) = {
    val UIMap = new Long2ObjectOpenHashMap[Array[(Long, Int)]]()
    val firstKeys = new LongOpenHashSet()

    def first(): Unit = {
      val random = new Random(System.currentTimeMillis())
      for (idx <- batchIdx) {
        val nbrsArray = new ArrayBuffer[(Long, Int)]()
        val len = indptr(idx + 1) - indptr(idx)
        var start = 0
        if (len > numSample)
          start = random.nextInt(len)
        val size = math.min(len, numSample)
        for (j <- 0 until size) {
          val jj = (j + start) % len + indptr(idx)
          val n = neighbors(jj)
          firstKeys.add(n)
          nbrsArray.append((n, types(jj)))
        }
        UIMap.put(keys(idx), nbrsArray.toArray)
      }
    }
    first()
    // make second order edge indexes
    val second = model.sampleNeighbors(firstKeys.toLongArray, numSample)
    for (idx <- batchIdx) {
      val items = UIMap.get(keys(idx))
      items.foreach { case (u, t) =>
        val nbrs = second.get(u)
        nbrs.foreach { nbr =>
          if (!index.containsKey(nbr)) {
            index.put(nbr, index.size())
          }
          srcs.add(index.get(keys(idx)))
          dsts.add(index.get(nbr))
          edgeTypes.add(t)
        }
      }
    }
    val edgeIndex = makeEdgeIndexFromSrcDst(srcs, dsts)
    val typesArray = edgeTypes.toLongArray
    (edgeIndex, typesArray)
  }

  /*
    Make edge Index for algorithms with edge types, for example R-GCN.
    There are two major differences.
    1. The returned structure includes an EdgeType array.
    2. We do not add self-loops since we do not know the types for self-loops. (Note that,
    we did not remove loops when loading edges.)
   */
  def makeEdgeIndex(batchIdx: Array[Int],
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    edgeTypes: LongArrayList,
                    batchKeys: LongOpenHashSet,
                    index: Long2IntOpenHashMap,
                    numSample: Int,
                    model: GNNPSModel): (Array[Long], Array[Long], Array[Long], Array[Long]) = {

    val (first, firstTypes, firstKeys) = makeFirstOrderEdgeIndex(
      batchIdx, keys, indptr, neighbors, types,
      srcs, dsts, edgeTypes, batchKeys, index, numSample)

    val (second, secondTypes) = makeSecondOrderEdgeIndex(batchKeys, firstKeys,
      // when edge has types, do not add self-loops due to the unknown of types for loops
      srcs, dsts, edgeTypes, index, model, numSample, edgeTypes == null)
    (first, firstTypes, second, secondTypes)
  }

  /*
    Make edge Index for graphs whose edges have different types. The key difference is that
    we cannot add self-loops for each source nodes since we cannot determine the types for
    added self-loops. Moreover, the neighbor sampling method is different. We cannot incur a
    inplace-shuffle for neighbors. We choose a random start point for sampling instead.
   */
  private
  def makeFirstOrderEdgeIndex(batchIdx: Array[Int],
                              keys: Array[Long],
                              indptr: Array[Int],
                              neighbors: Array[Long],
                              types: Array[Int],
                              srcs: LongArrayList,
                              dsts: LongArrayList,
                              edgeTypes: LongArrayList,
                              batchKeys: LongOpenHashSet,
                              index: Long2IntOpenHashMap,
                              numSample: Int): (Array[Long], Array[Long], LongOpenHashSet) = {
    val random = new Random(System.currentTimeMillis())
    val firstKeys = new LongOpenHashSet()
    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = index.get(key)
      // do not add self-loop since we cannot detect the type for self-loop edges
      val len = indptr(idx + 1) - indptr(idx)
      var start = 0
      if (len > numSample)
        start = random.nextInt(len)
      val size = math.min(len, numSample)
      for (j <- 0 until size) {
        val jj = (j + start) % len + indptr(idx)
        val n = neighbors(jj)
        if (!batchKeys.contains(n))
          firstKeys.add(n)
        if (!index.containsKey(n))
          index.put(n, index.size())
        srcs.add(keyIndex)
        dsts.add(index.get(n))
        edgeTypes.add(types(jj))
      }
    }
    (makeEdgeIndexFromSrcDst(srcs, dsts), edgeTypes.toLongArray, firstKeys)
  }

  /*
    Make edge Index for algorithms requires two-order neighbor graph.
    We return the firstKeys as a hashset, which contains the sources nodes for the
    two-order neighbor graph.
   */
  private
  def makeFirstOrderEdgeIndex(batchIdx: Array[Int],
                              keys: Array[Long],
                              indptr: Array[Int],
                              neighbors: Array[Long],
                              edgeDim: Int,
                              features: Array[IntFloatVector],
                              srcs: LongArrayList,
                              dsts: LongArrayList,
                              edgeFeatures: util.ArrayList[IntFloatVector],
                              batchKeys: LongOpenHashSet,
                              index: Long2IntOpenHashMap,
                              numSample: Int): ((Array[Long], Array[Float]), LongOpenHashSet) = {

    val firstKeys = new LongOpenHashSet()
    val set = new util.HashSet[Int]()
    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = index.get(key)
      srcs.add(keyIndex) // add self-loop
      dsts.add(keyIndex)
      edgeFeatures.add(new IntFloatVector(edgeDim, new IntFloatDenseVectorStorage(edgeDim)))
      val len = indptr(idx + 1) - indptr(idx)
      val rand = new util.Random()
      var indices: Array[Int] = null

      if (len > numSample) {
        while (set.size() < numSample) {
          val t = rand.nextInt(numSample)
          if (!set.contains(t)) {
            set.add(t)
          }
        }
        indices = set.toSet.toArray
        set.clear()
      } else {
        indices = (indptr(idx) until indptr(idx + 1)).toArray[Int]
      }

      indices.foreach{ j =>
        val n = neighbors(j)
        val edgeFeats = features(j)
        if (!batchKeys.contains(n))
          firstKeys.add(n)
        if (!index.containsKey(n))
          index.put(n, index.size())
        srcs.add(keyIndex)
        dsts.add(index.get(n))
        edgeFeatures.add(edgeFeats)
      }
    }

    (makeEdgeIndexAndFeaturesFromSrcDst(srcs, dsts, edgeFeatures, edgeDim), firstKeys)
  }

  /*
    Make edge Index for algorithms requires two-order neighbor graph.
    We return the firstKeys as a hashset, which contains the sources nodes for the
    two-order neighbor graph.
   */
  private
  def makeFirstOrderEdgeIndex(batchIdx: Array[Int],
                              keys: Array[Long],
                              indptr: Array[Int],
                              neighbors: Array[Long],
                              types: Array[Int],
                              edgeDim: Int,
                              features: Array[IntFloatVector],
                              srcs: LongArrayList,
                              dsts: LongArrayList,
                              edgeTypes: LongArrayList,
                              edgeFeatures: util.ArrayList[IntFloatVector],
                              batchKeys: LongOpenHashSet,
                              index: Long2IntOpenHashMap,
                              numSample: Int): ((Array[Long], Array[Float]), Array[Long], LongOpenHashSet) = {

    val firstKeys = new LongOpenHashSet()
    val set = new util.HashSet[Int]()
    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = index.get(key)
      // do not add self-loop since we cannot detect the type for self-loop edges
      val len = indptr(idx + 1) - indptr(idx)
      val rand = new util.Random()
      var indices: Array[Int] = null

      if (len > numSample) {
        while (set.size() < numSample) {
          val t = rand.nextInt(numSample)
          if (!set.contains(t)) {
            set.add(t)
          }
        }
        indices = set.toSet.toArray
        set.clear()
      } else {
        indices = (indptr(idx) until indptr(idx + 1)).toArray[Int]
      }

      indices.foreach{ j =>
        val n = neighbors(j)
        val edgeFeats = features(j)
        if (!batchKeys.contains(n))
          firstKeys.add(n)
        if (!index.containsKey(n))
          index.put(n, index.size())
        srcs.add(keyIndex)
        dsts.add(index.get(n))
        edgeFeatures.add(edgeFeats)
        edgeTypes.add(types(j))
      }
    }

    (makeEdgeIndexAndFeaturesFromSrcDst(srcs, dsts, edgeFeatures, edgeDim), edgeTypes.toLongArray, firstKeys)
  }

  private
  def makeSecondOrderEdgeIndex(batchKeys: LongOpenHashSet,
                               firstKeys: LongOpenHashSet,
                               srcs: LongArrayList,
                               dsts: LongArrayList,
                               types: LongArrayList,
                               index: Long2IntOpenHashMap,
                               model: GNNPSModel,
                               numSample: Int,
                               selfLoops: Boolean): (Array[Long], Array[Long]) = {
    val keys = firstKeys.toLongArray
    if (selfLoops) { // add self-loops for source nodes
      for (s <- keys) {
        val srcIndex = index.get(s)
        if (!batchKeys.contains(s)) { // for nodes in batchKeys, loops are already added
          srcs.add(srcIndex)
          dsts.add(srcIndex)
        }
      }
    }

    if (selfLoops) {
      val sampleNeighbors = model.sampleNeighbors(keys, numSample)
      val iter = sampleNeighbors.long2ObjectEntrySet().fastIterator()
      while (iter.hasNext) {
        val entry = iter.next()
        val key = entry.getLongKey
        val neighbors = entry.getValue
        for(nodeId <- neighbors)
        {
          if (!index.containsKey(nodeId)) {
            index.put(nodeId, index.size())
          }
          srcs.add(index.get(key))
          dsts.add(index.get(nodeId))
        }
      }
    } else {
      val sampleNeighborsWithTypes = model.sampleNeighborsWithType(keys, numSample, SampleType.NODE)
      val nodeNeighbors = sampleNeighborsWithTypes._1
      val nodeTypes = sampleNeighborsWithTypes._2
      val iter = nodeNeighbors.long2ObjectEntrySet().fastIterator()
      while (iter.hasNext) {
        val entry = iter.next()
        val key = entry.getLongKey
        val neighbors = entry.getValue
        val neighborsTypes = nodeTypes.get(key)
        for(i <- neighbors.indices)
        {
          if (!index.containsKey(neighbors(i))) {
            index.put(neighbors(i), index.size())
          }
          srcs.add(index.get(key))
          dsts.add(index.get(neighbors(i)))
          types.add(neighborsTypes(i))
        }
      }
    }

    val edgeIndex = makeEdgeIndexFromSrcDst(srcs, dsts)
    val typesArray = if (types == null) null else types.toLongArray
    (edgeIndex, typesArray)
  }

  private
  def makeSecondOrderEdgeIndex(batchKeys: LongOpenHashSet,
                               firstKeys: LongOpenHashSet,
                               srcs: LongArrayList,
                               dsts: LongArrayList,
                               features: util.ArrayList[IntFloatVector],
                               index: Long2IntOpenHashMap,
                               model: GNNPSModel,
                               numSample: Int,
                               edgeDim: Int): (Array[Long], Array[Float]) = {
    val (edgeIndex, _) = makeSecondOrderEdgeIndex(batchKeys, firstKeys, srcs, dsts,
      null, features, index, model, numSample, edgeDim, true)
    edgeIndex
  }

  private
  def makeSecondOrderEdgeIndex(batchKeys: LongOpenHashSet,
                               firstKeys: LongOpenHashSet,
                               srcs: LongArrayList,
                               dsts: LongArrayList,
                               types: LongArrayList,
                               features: util.ArrayList[IntFloatVector],
                               index: Long2IntOpenHashMap,
                               model: GNNPSModel,
                               numSample: Int,
                               edgeDim: Int,
                               selfLoops: Boolean): ((Array[Long], Array[Float]), Array[Long]) = {
    val keys = firstKeys.toLongArray
    if (selfLoops) {
      for (s <- keys) {
        val srcIndex = index.get(s)
        if (!batchKeys.contains(s)) { // for nodes in batchKeys, loops are already added
          srcs.add(srcIndex)
          dsts.add(srcIndex)
          features.add(new IntFloatVector(edgeDim, new IntFloatDenseVectorStorage(edgeDim)))
        }
      }
    }


    val neighborsWithEdgeFeat = model.sampleEdgeFeatures(keys, numSample)
    val nodeNeighbors = neighborsWithEdgeFeat._1
    val nodeEdgeFeats = neighborsWithEdgeFeat._2
    val iter = nodeNeighbors.long2ObjectEntrySet().fastIterator()
    while (iter.hasNext) {
      val entry = iter.next()
      val key = entry.getLongKey
      val neighbors = entry.getValue
      val edgeFeats = nodeEdgeFeats.get(key)
      for(i <- neighbors.indices)
      {
        if (!index.containsKey(neighbors(i))) {
          index.put(neighbors(i), index.size())
        }
        srcs.add(index.get(key))
        dsts.add(index.get(neighbors(i)))
        features.add(edgeFeats(i))
      }
    }

    val edgeIndex = makeEdgeIndexAndFeaturesFromSrcDst(srcs, dsts, features, edgeDim)
    val typesArray = if (types == null) null else types.toLongArray
    (edgeIndex, typesArray)
  }

  def makeEdgeIndexFromSrcDst(srcs: LongArrayList, dsts: LongArrayList): Array[Long] = {
    assert(srcs.size() == dsts.size())
    val edgeIndex = new Array[Long](srcs.size() * 2)
    val size = srcs.size()
    for (i <- 0 until size) {
      edgeIndex(i) = srcs.getLong(i)
      edgeIndex(i + size) = dsts.getLong(i)
    }
    edgeIndex
  }

  def makeEdgeIndexAndFeaturesFromSrcDst(srcs: LongArrayList, dsts: LongArrayList,
                                         edgeFeatures: util.ArrayList[IntFloatVector],
                                         edgeDim: Int): (Array[Long], Array[Float]) = {
    assert(srcs.size() == dsts.size())
    val size = srcs.size()
    val edgeIndex = new Array[Long](srcs.size() * 2)
    val x = new Array[Float](size * edgeDim)
    for (i <- 0 until size) {
      edgeIndex(i) = srcs.getLong(i)
      edgeIndex(i + size) = dsts.getLong(i)
      MakeFeature.makeFeature(i * edgeDim, edgeFeatures.get(i), x)
    }
    (edgeIndex, x)
  }

  /*
    Make edge Index for nodes without out-edges, we only add self-loops for these nodes
   */
  def makeEdgeIndex(keys: Array[Long],
                    index: Long2IntOpenHashMap): Array[Long] = {
    val edgeIndex = new Array[Long](keys.length * 2)
    for (i <- keys.indices) {
      edgeIndex(i) = index.get(keys(i))
      edgeIndex(i + keys.length) = index.get(keys(i))
    }
    edgeIndex
  }

}
