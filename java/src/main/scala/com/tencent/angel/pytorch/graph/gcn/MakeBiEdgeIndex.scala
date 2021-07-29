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

import com.tencent.angel.graph.client.psf.sample.sampleneighbor.SampleType
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongArrays, LongOpenHashSet}

object MakeBiEdgeIndex {

  /*
    Make edge Index for algorithms on Bipartite graphs: Bipartite graphsage
    edges: u-i-u
   */
  def makeEdgeIndex(batchIdx: Array[Int],
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    edgeTypes: Array[Int],
                    itemTypes: Array[Int],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    edgeTypesList: IntArrayList,
                    itemTypesList: IntArrayList,
                    srcIndex: Long2IntOpenHashMap,
                    dstIndex: Long2IntOpenHashMap,
                    userNumSample: Int,
                    itemNumSample: Int,
                    model: BiSAGEPSModel,
                    useSecondOrder: Boolean,
                    graphType: Int,
                    hasEdgeType: Boolean,
                    hasNodeType: Boolean): (Array[Long], Array[Long], Array[Long], Array[Int], Array[Int],
    Array[Int], Array[Int], Array[Int]) = {

    val sampleNums_first = if (graphType==0) userNumSample else itemNumSample
    val sampleNums_second = if (graphType==0) itemNumSample else userNumSample

    if (!useSecondOrder) {
      val (edgeIndex, eTypes, iTypes) = makeEdgeIndex(batchIdx, keys, indptr, neighbors, edgeTypes, itemTypes,
        srcs, dsts, edgeTypesList, itemTypesList, srcIndex, dstIndex, sampleNums_first, hasEdgeType, hasNodeType)
      (edgeIndex, null, null, eTypes, iTypes, null, null, null)
    } else {
      val (first, firstKeys, eTypes, iTypes) = makeFirstOrderEdgeIndex(
        batchIdx, keys, indptr, neighbors, edgeTypes, itemTypes, srcs, dsts, edgeTypesList, itemTypesList,
        srcIndex, dstIndex, sampleNums_first, hasEdgeType, hasNodeType)

      if (firstKeys.size() == 0) // for nodes without out-edges
        return (first, first, first, eTypes, iTypes, eTypes, eTypes, iTypes)

      val srcs_ = new LongArrayList()
      val dsts_ = new LongArrayList()
      val edgeTypesList_ = new IntArrayList()
      val (second, secondKeys, eSecondTypes, _) = makeSecondOrderEdgeIndex(firstKeys, srcs_, dsts_, edgeTypesList_,
        itemTypesList, dstIndex, srcIndex, model, sampleNums_second, 1 - graphType, hasEdgeType, hasNodeType)

      val (second_, _, eSecondTypes_, iSecondTypes) = makeSecondOrderEdgeIndex(secondKeys, srcs, dsts, edgeTypesList,
        itemTypesList, srcIndex, dstIndex, model, sampleNums_first, graphType, hasEdgeType, hasNodeType)

      (first, second, second_, eTypes, iTypes, eSecondTypes, eSecondTypes_, iSecondTypes)
    }
  }

  private
  def makeEdgeIndex(batchIdx: Array[Int],
                    keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    edgeTypes: Array[Int],
                    itemTypes: Array[Int],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    edgeTypesList: IntArrayList,
                    itemTypesList: IntArrayList,
                    srcIndex: Long2IntOpenHashMap,
                    dstIndex: Long2IntOpenHashMap,
                    numSample: Int,
                    hasEdgeType: Boolean,
                    hasNodeType: Boolean): (Array[Long], Array[Int], Array[Int])= {
    val random = new Random(System.currentTimeMillis())

    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = srcIndex.get(key)
      // do not add self-loop since we cannot detect the type for self-loop edges
      val len = indptr(idx + 1) - indptr(idx)
      var start = 0
      if (len > numSample)
        start = random.nextInt(len)
      val size = math.min(len, numSample)
      for (j <- 0 until size) {
        val jj = (j + start) % len + indptr(idx)
        val n = neighbors(jj)
        if (!dstIndex.containsKey(n))
          dstIndex.put(n, dstIndex.size())
        srcs.add(keyIndex)
        dsts.add(dstIndex.get(n))
        if (hasEdgeType) edgeTypesList.add(edgeTypes(jj))
        if (hasNodeType) itemTypesList.add(itemTypes(jj))
      }
    }
    val eTypes = if (hasEdgeType) edgeTypesList.toIntArray() else null
    val iTypes = if (hasNodeType) itemTypesList.toIntArray() else null

    (makeEdgeIndexFromSrcDst(srcs, dsts), eTypes, iTypes)
  }

  /*
    Make edge Index for algorithms on Bipartite graphs: IGMC
    edges: u-i, i-u
   */
  def makeEdgeIndex(batchIdx: Array[(Int, Int)],
                    keys: Array[Long],
                    itemKeys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    edgeTypes: IntArrayList,
                    srcIndex: Long2IntOpenHashMap,
                    dstIndex: Long2IntOpenHashMap,
                    numSample: Int,
                    model: BiSAGEPSModel,
                    useSecondOrder: Boolean,
                    graphType: Int,
                    hasEdgeType: Boolean): (Array[Long], Array[Int]) = {
    if (!useSecondOrder) {
      //sample total neighbors of u
      val (_, iKey, _, iAloneKey) = makeEdgeIndex(batchIdx, keys, itemKeys, indptr, neighbors, types, srcs, dsts,
        edgeTypes, srcIndex, dstIndex, numSample, hasEdgeType)

      //sample total neighbors of i
      val (_, uKey, _) = makeVUEdgeIndex(batchIdx.map(f => itemKeys(f._2)), batchIdx.map(f => keys(f._1)),
        null, dsts, srcs, edgeTypes, dstIndex, srcIndex, model, numSample, 1 - graphType, hasEdgeType)

      //remove labeled edge from uKey and iKey
      batchIdx.foreach { idx =>
        if (uKey.contains(keys(idx._1))) {
          uKey.remove(keys(idx._1))
        }

        if (iKey.contains(itemKeys(idx._2))) {
          iKey.remove(itemKeys(idx._2))
        }
      }

      //sample total edges between uKey and iKey
      if (uKey.size > iKey.size && iKey.size() != 0) {
        for (k <- uKey.toLongArray) {
          if (!srcIndex.containsKey(k)) {
            srcIndex.put(k, srcIndex.size())
          }
        }
        makeVUEdgeIndex(uKey.toLongArray, null, iKey.toLongArray, srcs, dsts,
          edgeTypes, srcIndex, dstIndex, model, numSample, graphType, hasEdgeType)
      } else if (uKey.size < iKey.size && uKey.size() != 0) {
        for (k <- iKey.toLongArray) {
          if (!dstIndex.containsKey(k)) {
            dstIndex.put(k, dstIndex.size())
          }
        }
        makeVUEdgeIndex(iKey.toLongArray, null, uKey.toLongArray, dsts, srcs,
          edgeTypes, dstIndex, srcIndex, model, numSample, 1 - graphType, hasEdgeType)
      }

      if (iAloneKey.size() > 0) {
        val (_, secondKeys, _, _) = makeSecondOrderEdgeIndex(iAloneKey, dsts, srcs, edgeTypes,
          null, dstIndex, srcIndex, model, 5, 1 - graphType, hasEdgeType, false)

        val (_, _, _, _) = makeSecondOrderEdgeIndex(secondKeys, srcs, dsts, edgeTypes,
          null, srcIndex, dstIndex, model, 5, graphType, hasEdgeType, false)
      }

      //make u-i edge index by srcs and dsts
      val index_ = makeEdgeIndexFromSrcDst(srcs, dsts)

      (index_, edgeTypes.toIntArray)
    } else {
      (null, null)
    }
  }

  //for IGMC  sample neighbors of keys and remove labeled edge, whose neighbor's id in itemKeys
  def makeEdgeIndex(batchIdx: Array[(Int, Int)],
                    keys: Array[Long],
                    itemKeys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    types: Array[Int],
                    srcs: LongArrayList,
                    dsts: LongArrayList,
                    edgeTypes: IntArrayList,
                    srcIndex: Long2IntOpenHashMap,
                    dstIndex: Long2IntOpenHashMap,
                    numSample: Int,
                    hasEdgeType: Boolean): (Array[Long], LongOpenHashSet, Array[Int], LongOpenHashSet) = {
    val random = new Random(System.currentTimeMillis())
    val firstKeys = new LongOpenHashSet()
    val firstAloneKeys = new LongOpenHashSet()

    for (idxs <- batchIdx) {
      val idx = idxs._1
      val key = keys(idx)
      val keyIndex = srcIndex.get(key)
      // do not add self-loop since we cannot detect the type for self-loop edges
      val len = indptr(idx + 1) - indptr(idx)
      var start = 0
      if (numSample > 0 && len > numSample)
        start = random.nextInt(len)
      val size = if (numSample > 0) math.min(len, numSample) else len
      for (j <- 0 until size) {
        val jj = (j + start) % len + indptr(idx)
        val n = neighbors(jj)
        if (len != 1) firstKeys.add(n) else firstAloneKeys.add(n)
        if (len == 1 || n != itemKeys(idxs._2)) {
          //remove labeled edge
          if (!dstIndex.containsKey(n)) {
            dstIndex.put(n, dstIndex.size())
          }

          srcs.add(keyIndex)
          dsts.add(dstIndex.get(n))
          if (hasEdgeType)
            edgeTypes.add(types(jj))
        }
      }
    }
    val iTypes = if (hasEdgeType) edgeTypes.toIntArray() else null

    (makeEdgeIndexFromSrcDst(srcs, dsts), firstKeys, iTypes, firstAloneKeys)
  }

  /*
    Make edge Index for algorithms requires two-order neighbor graph.
    We return the firstKeys as a hashset, which contains the sources nodes for the
    two-order neighbor graph.
   */
  def makeFirstOrderEdgeIndex(batchIdx: Array[Int],
                              keys: Array[Long],
                              indptr: Array[Int],
                              neighbors: Array[Long],
                              srcs: LongArrayList,
                              dsts: LongArrayList,
                              srcIndex: Long2IntOpenHashMap,
                              dstIndex: Long2IntOpenHashMap,
                              numSample: Int): (Array[Long], LongOpenHashSet) = {
    val random = new Random(System.currentTimeMillis())
    val firstKeys = new LongOpenHashSet()
    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = srcIndex.get(key)
      val len = indptr(idx + 1) - indptr(idx)
      if (len > numSample)
        LongArrays.shuffle(neighbors, indptr(idx), indptr(idx + 1), random)
      val size = math.min(len, numSample)
      var j = indptr(idx)
      while (j < indptr(idx) + size) {
        val n = neighbors(j)
        firstKeys.add(n)//todo???
        if (!dstIndex.containsKey(n))
          dstIndex.put(n, dstIndex.size())
        srcs.add(keyIndex)
        dsts.add(dstIndex.get(n))
        j += 1
      }
    }

    (makeEdgeIndexFromSrcDst(srcs, dsts), firstKeys)
  }

  def makeFirstOrderEdgeIndex(batchIdx: Array[Int],
                              keys: Array[Long],
                              indptr: Array[Int],
                              neighbors: Array[Long],
                              edgeTypes: Array[Int],
                              itemTypes: Array[Int],
                              srcs: LongArrayList,
                              dsts: LongArrayList,
                              edgeTypesList: IntArrayList,
                              itemTypesList: IntArrayList,
                              srcIndex: Long2IntOpenHashMap,
                              dstIndex: Long2IntOpenHashMap,
                              numSample: Int,
                              hasEdgeType: Boolean,
                              hasNodeType: Boolean): (Array[Long], LongOpenHashSet, Array[Int], Array[Int]) = {
    val random = new Random(System.currentTimeMillis())
    val firstKeys = new LongOpenHashSet()

    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = srcIndex.get(key)
      // do not add self-loop since we cannot detect the type for self-loop edges
      val len = indptr(idx + 1) - indptr(idx)
      var start = 0
      if (len > numSample)
        start = random.nextInt(len)
      val size = math.min(len, numSample)
      for (j <- 0 until size) {
        val jj = (j + start) % len + indptr(idx)
        val n = neighbors(jj)
        if (!dstIndex.containsKey(n))
          dstIndex.put(n, dstIndex.size())
        firstKeys.add(n)
        srcs.add(keyIndex)
        dsts.add(dstIndex.get(n))
        if (hasEdgeType) edgeTypesList.add(edgeTypes(jj))
        if (hasNodeType) itemTypesList.add(itemTypes(jj))
      }
    }
    val eTypes = if (hasEdgeType && itemTypesList != null) edgeTypesList.toIntArray else null
    val iTypes = if (hasNodeType && itemTypesList != null) itemTypesList.toIntArray else null
    (makeEdgeIndexFromSrcDst(srcs, dsts), firstKeys, eTypes, iTypes)
  }

  // make second order edges and return the dst nodes
  def makeSecondOrderEdgeIndex(firstKeys: LongOpenHashSet,
                               srcs: LongArrayList,
                               dsts: LongArrayList,
                               srcIndex: Long2IntOpenHashMap,
                               dstIndex: Long2IntOpenHashMap,
                               model: BiSAGEPSModel,
                               numSample: Int,
                               graphType: Int): (Array[Long], LongOpenHashSet) = {
    val seconds = model.sampleNeighbors(firstKeys.toLongArray, numSample, graphType)
    val it = seconds.long2ObjectEntrySet().fastIterator()
    val secondKeys = new LongOpenHashSet()
    while (it.hasNext) {
      val entry = it.next()
      val (s, ds) = (entry.getLongKey, entry.getValue)
      val src_ = srcIndex.get(s) // s must in index
      for (d <- ds) {
        if (!dstIndex.containsKey(d))
          dstIndex.put(d, dstIndex.size())
        secondKeys.add(d)
        srcs.add(src_)
        dsts.add(dstIndex.get(d))
      }
    }

    (makeEdgeIndexFromSrcDst(srcs, dsts), secondKeys)
  }

  def makeSecondOrderEdgeIndex(firstKeys: LongOpenHashSet,
                               srcs: LongArrayList,
                               dsts: LongArrayList,
                               edgeTypesList: IntArrayList,
                               itemTypesList: IntArrayList,
                               srcIndex: Long2IntOpenHashMap,
                               dstIndex: Long2IntOpenHashMap,
                               model: BiSAGEPSModel,
                               numSample: Int,
                               graphType: Int,
                               hasEdgeType: Boolean,
                               hasNodeType: Boolean): (Array[Long], LongOpenHashSet, Array[Int], Array[Int]) = {

    val secondKeys = new LongOpenHashSet()
    val edgeTypes = if (hasEdgeType) edgeTypesList else null
    val itemTypes = if (graphType == 0 && hasNodeType) itemTypesList else null

    if (edgeTypes != null && itemTypes != null) {
      val sampleNeighborsWithTypes = model.sampleNeighborsWithType(firstKeys.toLongArray, numSample,
        SampleType.NODE_AND_EDGE, graphType)
      val nodeNeighbors = sampleNeighborsWithTypes._1
      val sampleNodeTypes = sampleNeighborsWithTypes._2
      val sampleEdgeTypes = sampleNeighborsWithTypes._3
      val iter = nodeNeighbors.long2ObjectEntrySet().fastIterator()
      while (iter.hasNext) {
        val entry = iter.next()
        val key = entry.getLongKey
        val neighbors = entry.getValue
        val neighborsTypes = sampleNodeTypes.get(key)
        val neighborEdgesTypes = sampleEdgeTypes.get(key)
        for(i <- neighbors.indices)
        {
          if (!dstIndex.containsKey(neighbors(i))) {
            dstIndex.put(neighbors(i), dstIndex.size())
          }
          secondKeys.add(neighbors(i))
          srcs.add(srcIndex.get(key))
          dsts.add(dstIndex.get(neighbors(i)))
          itemTypes.add(neighborsTypes(i))
          edgeTypes.add(neighborEdgesTypes(i))
        }
      }
    } else if (itemTypes != null) {
      val sampleNeighborsWithTypes = model.sampleNeighborsWithType(firstKeys.toLongArray, numSample, SampleType.NODE, graphType)
      val nodeNeighbors = sampleNeighborsWithTypes._1
      val sampleNodeTypes = sampleNeighborsWithTypes._2
      val iter = nodeNeighbors.long2ObjectEntrySet().fastIterator()
      while (iter.hasNext) {
        val entry = iter.next()
        val key = entry.getLongKey
        val neighbors = entry.getValue
        val neighborsTypes = sampleNodeTypes.get(key)
        for(i <- neighbors.indices)
        {
          if (!dstIndex.containsKey(neighbors(i))) {
            dstIndex.put(neighbors(i), dstIndex.size())
          }
          secondKeys.add(neighbors(i))
          srcs.add(srcIndex.get(key))
          dsts.add(dstIndex.get(neighbors(i)))
          itemTypes.add(neighborsTypes(i))
        }
      }
    } else if (edgeTypes != null) {
      val sampleNeighborsWithTypes = model.sampleNeighborsWithType(firstKeys.toLongArray, numSample,
        SampleType.EDGE, graphType)
      val nodeNeighbors = sampleNeighborsWithTypes._1
      val sampleEdgeTypes = sampleNeighborsWithTypes._3
      val iter = nodeNeighbors.long2ObjectEntrySet().fastIterator()
      while (iter.hasNext) {
        val entry = iter.next()
        val key = entry.getLongKey
        val neighbors = entry.getValue
        val neighborEdgesTypes = sampleEdgeTypes.get(key)
        for(i <- neighbors.indices)
        {
          if (!dstIndex.containsKey(neighbors(i))) {
            dstIndex.put(neighbors(i), dstIndex.size())
          }
          secondKeys.add(neighbors(i))
          srcs.add(srcIndex.get(key))
          dsts.add(dstIndex.get(neighbors(i)))
          edgeTypes.add(neighborEdgesTypes(i))
        }
      }
    } else {
      val sampleNeighbors = model.sampleNeighbors(firstKeys.toLongArray, numSample, graphType)
      val iter = sampleNeighbors.long2ObjectEntrySet().fastIterator()
      while (iter.hasNext) {
        val entry = iter.next()
        val key = entry.getLongKey
        val neighbors = entry.getValue
        for(nodeId <- neighbors)
        {
          if (!dstIndex.containsKey(nodeId)) {
            dstIndex.put(nodeId, dstIndex.size())
          }
          secondKeys.add(nodeId)
          srcs.add(srcIndex.get(key))
          dsts.add(dstIndex.get(nodeId))
        }
      }
    }

    val eTypes = if (edgeTypes != null) edgeTypes.toIntArray else null
    val iTypes = if (graphType == 0 && itemTypes != null) itemTypes.toIntArray else null
    (makeEdgeIndexFromSrcDst(srcs, dsts), secondKeys, eTypes, iTypes)
  }

  //for IGMC sample v-u, u-v from ps,
  //and add some filter pattern, such as: without some neighbors or contain some neighbors
  def makeVUEdgeIndex(firstKeys: Array[Long],
                      withoutNeighKeys: Array[Long],
                      containNeighKeys: Array[Long],
                      srcs: LongArrayList,
                      dsts: LongArrayList,
                      edgeTypes: IntArrayList,
                      srcIndex: Long2IntOpenHashMap,
                      dstIndex: Long2IntOpenHashMap,
                      model: BiSAGEPSModel,
                      numSample: Int,
                      graphType: Int,
                      hasEdgeType: Boolean): (Array[Long], LongOpenHashSet, Array[Int]) = {

    val secondKeys = new LongOpenHashSet()
    val sampleNeighborsWithTypeWithFilter = model.sampleNeighborsWithTypeWithFilter(firstKeys, withoutNeighKeys,
      containNeighKeys, numSample, SampleType.EDGE, graphType)
    val nodeNeighbors = sampleNeighborsWithTypeWithFilter._1
    val sampleEdgeTypes = sampleNeighborsWithTypeWithFilter._3
    val iter = nodeNeighbors.long2ObjectEntrySet().fastIterator()
    while (iter.hasNext) {
      val entry = iter.next()
      val key = entry.getLongKey
      val neighbors = entry.getValue
      val eTypes = sampleEdgeTypes.get(key)
      for(i <- neighbors.indices)
      {
        if (!dstIndex.containsKey(neighbors(i))) {
          dstIndex.put(neighbors(i), dstIndex.size())
        }
        srcs.add(srcIndex.get(key))
        dsts.add(dstIndex.get(neighbors(i)))
        edgeTypes.add(eTypes(i))
      }
    }
    (makeEdgeIndexFromSrcDst(srcs, dsts), secondKeys, edgeTypes.toIntArray)
  }

  def makeAllEdgeIndex(batchIdx: Array[Int],
                       keys: Array[Long],
                       indptr: Array[Int],
                       neighbors: Array[Long],
                       srcs: LongArrayList,
                       dsts: LongArrayList,
                       srcIndex: Long2IntOpenHashMap,
                       dstIndex: Long2IntOpenHashMap): LongOpenHashSet = {
    val random = new Random(System.currentTimeMillis())
    val firstKeys = new LongOpenHashSet()
    for (idx <- batchIdx) {
      val key = keys(idx)
      val keyIndex = srcIndex.get(key)
      val len = indptr(idx + 1) - indptr(idx)
      val size = len
      var j = indptr(idx)
      while (j < indptr(idx) + size) {
        val n = neighbors(j)
        firstKeys.add(n)
        if (!dstIndex.containsKey(n))
          dstIndex.put(n, dstIndex.size())
        j += 1
      }
    }
    firstKeys
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