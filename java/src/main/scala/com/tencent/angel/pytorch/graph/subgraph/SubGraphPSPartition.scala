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
package com.tencent.angel.pytorch.graph.subgraph

import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.{LongArrayList, LongOpenHashSet}

import scala.collection.mutable.ArrayBuffer

private[subgraph]
class SubGraphPSPartition(index: Int,
                          keys: Array[Long],
                          indptr: Array[Int],
                          neighbors: Array[Long]) extends Serializable {

  def init(model: SubGraphPSModel, numBatch: Int): Int = {
    model.initNeighbors(keys, indptr, neighbors, numBatch)
    0
  }

  def sample(model: SubGraphPSModel): Unit = {
    val batchKeys = new LongOpenHashSet(keys.size)
    keys.foreach(k => batchKeys.add(k))
    val firstKeys = new LongOpenHashSet()

    val flags = model.readNodes(keys.clone())
    for (idx <- keys.indices) {
      if (flags.get(keys(idx)) > 0) {
        // node with label
        var j = indptr(idx)
        while (j < indptr(idx + 1)) {
          val n = neighbors(j)
          if (!batchKeys.contains(n)) {
            firstKeys.add(n)
            batchKeys.add(n)
          }
          j += 1
        }
      }
    }

    // now batchKeys contain all the srcs we needed
    model.updateSrcKeys(batchKeys.toLongArray)

    // get neighbors of first keys
    val seconds = model.sampleNeighbors(firstKeys.toLongArray, -1)
    val it = seconds.long2ObjectEntrySet().fastIterator()
    while (it.hasNext) {
      val entry = it.next()
      val ns = entry.getValue
      for (k <- ns.indices) {
        if (!batchKeys.contains(ns(k)))
          batchKeys.add(ns(k))
      }
    }

    // now batchKeys contain all the nodes we needed
    model.setNodes(batchKeys.toLongArray)

  }

  def twoOrderEdges(model: SubGraphPSModel): Iterator[(Long, Long)] = {
    val flags = model.readSrcKeys(keys.clone())
    val edges = new ArrayBuffer[(Long, Long)]()
    for (idx <- keys.indices) {
      if (flags.get(keys(idx)) > 0) {
        var j = indptr(idx)
        while (j < indptr(idx + 1)) {
          edges.append((keys(idx), neighbors(j)))
          j += 1
        }
      }
    }
    edges.iterator
  }
}

private[subgraph]
object SubGraphPSPartition {
  def apply(index: Int, iterator: Iterator[(Long, Iterable[Long])]): SubGraphPSPartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbors = new LongArrayList()

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      ns.foreach(n => neighbors.add(n))
      indptr.add(neighbors.size())
      keys.add(node)
    }

    new SubGraphPSPartition(index, keys.toLongArray,
      indptr.toIntArray, neighbors.toLongArray)
  }
}
