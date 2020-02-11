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

import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.vector.{IntFloatVector, LongFloatVector}
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList}

import scala.collection.mutable.ArrayBuffer

private[gcn]
class GraphAdjPartition(index: Int,
                        keys: Array[Long],
                        indptr: Array[Int],
                        neighbours: Array[Long]) extends Serializable {

  def init(model: GNNPSModel, numBatch: Int): Int = {
    // init adjacent table on servers
    model.initNeighbors(keys, indptr, neighbours, numBatch)
    0
  }

  def splitTrainTest(model: GNNPSModel, testRatio: Float): (Array[Int], Array[Int], Array[Float], Array[Float]) = {
    val myLabels = model.readLabels2(keys)
    val it = myLabels.getStorage.entryIterator()
    val size = myLabels.size().toInt
    val idxHasLabels = new Array[Int](size)
    val labels = new Array[Float](size)
    val position = new Long2IntOpenHashMap(keys.size)
    for (idx <- keys.indices)
      position.put(keys(idx), idx)
    var idx = 0
    while (it.hasNext) {
      val entry = it.next()
      val (node, label) = (entry.getLongKey, entry.getFloatValue)
      idxHasLabels(idx) = position.get(node)
      labels(idx) = label
      idx += 1
    }

    val splitPoint = (size * (1 - testRatio)).toInt
    val (trainIdx, testIdx) = idxHasLabels.splitAt(splitPoint)
    val (trainLabels, testLabels) = labels.splitAt(splitPoint)
    (trainIdx, testIdx, trainLabels, testLabels)
  }

  def getTrainTest(model: GNNPSModel): (Array[Int], Array[Int], Array[Float], Array[Float]) = {
    val trainLabels = model.readLabels2(keys)
    val testLabels = model.readTestLabels(keys)
    val position = new Long2IntOpenHashMap(keys.size)
    for (idx <- keys.indices)
      position.put(keys(idx), idx)

    def getIdxAndLabels(labels: LongFloatVector): (Array[Int], Array[Float]) = {
      val it = labels.getStorage.entryIterator()
      val indices = new Array[Int](labels.size().toInt)
      val labelArray = new Array[Float](labels.size().toInt)
      var idx = 0
      while (it.hasNext) {
        val entry = it.next()
        val (node, label) = (entry.getLongKey, entry.getFloatValue)
        labelArray(idx) = label
        indices(idx) = position.get(node)
        idx += 1
      }
      (indices, labelArray)
    }

    val (trainIdx, trainLabelArray) = getIdxAndLabels(trainLabels)
    val (testIdx, testLabelArray) = getIdxAndLabels(testLabels)
    (trainIdx, testIdx, trainLabelArray, testLabelArray)
  }

  def toSemiGCNPartition(model: GNNPSModel, torchModelPath: String, testRatio: Float): GCNPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = splitTrainTest(model, testRatio)
    new GCNPartition(index, keys, indptr, neighbours,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath)
  }

  def toSemiGCNPartition(model: GNNPSModel, torchModelPath: String): GCNPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = getTrainTest(model)
    new GCNPartition(index, keys, indptr, neighbours,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath)
  }

  def toMiniBatchDGIPartition(model:GNNPSModel, torchModelPath: String, useSecondOrder: Boolean): DGIPartition = {
    new DGIPartition(index, keys, indptr, neighbours, torchModelPath, useSecondOrder)
  }
}

private[gcn]
class GraphAdjTypePartition(index: Int,
                            keys: Array[Long],
                            indptr: Array[Int],
                            neighbors: Array[Long],
                            types: Array[Int]) extends GraphAdjPartition(index, keys, indptr, neighbors) {
  override
  def init(model: GNNPSModel, numBatch: Int): Int = {
    model.initNeighbors(keys, indptr, neighbors, types, numBatch)
    0
  }

  def toSemiRGCNPartition(model: GNNPSModel, torchModelPath: String, testRatio: Float): RGCNPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = splitTrainTest(model, testRatio)
    new RGCNPartition(index, keys, indptr, neighbors, types,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath)
  }
}

private[gcn]
class NodeFeaturePartition(index: Int,
                           keys: Array[Long],
                           features: Array[IntFloatVector]) extends Serializable {
  def init(model: GNNPSModel, numBatch: Int): Unit =
    model.initNodeFeatures(keys, features, numBatch)
}

private[gcn]
class NodeLabelPartition(index: Int, labels: LongFloatVector) extends Serializable {
  def init(model: GNNPSModel): Unit =
    model.setLabels(labels)

  def initTestLabels(model: GNNPSModel): Unit =
    model.setTestLabels(labels)
}


private[gcn]
object GraphAdjPartition {
  def apply(index: Int, iterator: Iterator[(Long, Iterable[Long])]): GraphAdjPartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbours = new LongArrayList()

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      ns.foreach(n => neighbours.add(n))
      indptr.add(neighbours.size())
      keys.add(node)
    }

    new GraphAdjPartition(index, keys.toLongArray,
      indptr.toIntArray, neighbours.toLongArray)
  }
}

object GraphAdjTypePartition {
  def apply(index: Int, iterator: Iterator[(Long, Iterable[(Long, Int)])]): GraphAdjTypePartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbors = new LongArrayList()
    val types = new IntArrayList()

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      for ((n, t) <- ns) {
        neighbors.add(n)
        types.add(t)
      }
      indptr.add(neighbors.size())
      keys.add(node)
    }

    new GraphAdjTypePartition(index, keys.toLongArray,
      indptr.toIntArray,
      neighbors.toLongArray,
      types.toIntArray)
  }
}

private[gcn]
object NodeLabelPartition {
  def apply(index: Int, iterator: Iterator[(Long, Float)], dim: Long): NodeLabelPartition = {
    val labels = VFactory.sparseLongKeyFloatVector(dim)
    while (iterator.hasNext) {
      val entry = iterator.next()
      labels.set(entry._1, entry._2)
    }
    new NodeLabelPartition(index, labels)
  }
}

private[gcn]
object NodeFeaturePartition {
  def apply(index: Int, iterator: Iterator[(Long, IntFloatVector)]): NodeFeaturePartition = {
    val keys = new LongArrayList()
    val features = new ArrayBuffer[IntFloatVector]()
    while (iterator.hasNext) {
      val entry = iterator.next()
      keys.add(entry._1)
      features.append(entry._2)
    }
    new NodeFeaturePartition(index, keys.toLongArray, features.toArray)
  }
}


