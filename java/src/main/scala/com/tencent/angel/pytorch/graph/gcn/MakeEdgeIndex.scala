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

import java.util.Random

import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongArrays, LongOpenHashSet}

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

  //  private
  //  def makeSecondOrderEdgeIndex0(batchKeys: LongOpenHashSet,
  //                               firstKeys: LongOpenHashSet,
  //                               srcs: LongArrayList,
  //                               dsts: LongArrayList,
  //                               index: Long2IntOpenHashMap,
  //                               model: GNNPSModel,
  //                               numSample: Int): Array[Long] = {
  //    val seconds = model.sampleNeighbors(firstKeys.toLongArray, numSample)
  //    val it = seconds.long2ObjectEntrySet().fastIterator()
  //    while (it.hasNext) {
  //      val entry = it.next()
  //      val (s, ds) = (entry.getLongKey, entry.getValue)
  //      val srcIndex = index.get(s) // s must in index
  //      if (!batchKeys.contains(s)) {
  //        srcs.add(srcIndex) // add self-loop, for node in batchKeys, the loop is already been added.
  //        dsts.add(srcIndex)
  //      }
  //      for (d <- ds) {
  //        if (!index.containsKey(d))
  //          index.put(d, index.size())
  //        srcs.add(srcIndex)
  //        dsts.add(index.get(d))
  //      }
  //    }
  //
  //    makeEdgeIndexFromSrcDst(srcs, dsts)
  //  }

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

    model.sampleNeighbors(keys, numSample, index, srcs, dsts, types)
    val edgeIndex = makeEdgeIndexFromSrcDst(srcs, dsts)
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

}
