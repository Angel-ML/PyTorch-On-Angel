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
  def makeSecondOrderEdgeIndex(batchKeys: LongOpenHashSet,
                               firstKeys: LongOpenHashSet,
                               srcs: LongArrayList,
                               dsts: LongArrayList,
                               index: Long2IntOpenHashMap,
                               model: GNNPSModel,
                               numSample: Int): Array[Long] = {
    val seconds = model.sampleNeighbors(firstKeys.toLongArray, numSample)
    val it = seconds.long2ObjectEntrySet().fastIterator()
    while (it.hasNext) {
      val entry = it.next()
      val (s, ds) = (entry.getLongKey, entry.getValue)
      val srcIndex = index.get(s) // s must in index
      if (!batchKeys.contains(s)) {
        srcs.add(srcIndex) // add self-loop, for node in batchKeys, the loop is already been added.
        dsts.add(srcIndex)
      }
      for (d <- ds) {
        if (!index.containsKey(d))
          index.put(d, index.size())
        srcs.add(srcIndex)
        dsts.add(index.get(d))
      }
    }

    makeEdgeIndexFromSrcDst(srcs, dsts)
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
