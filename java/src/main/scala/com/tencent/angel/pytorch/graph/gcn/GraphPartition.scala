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
import com.tencent.angel.ml.math2.storage.IntFloatDenseVectorStorage
import com.tencent.angel.ml.math2.vector.{IntFloatVector, LongFloatVector}
import com.tencent.angel.pytorch.graph.gcn.hetAttention.HANPartition
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList}
import it.unimi.dsi.fastutil.floats.FloatArrayList

import scala.util.Random
import scala.collection.mutable.ArrayBuffer

private[gcn]
class GraphAdjPartition(index: Int,
                        keys: Array[Long],
                        indptr: Array[Int],
                        neighbours: Array[Long]) extends Serializable {
  def numNodes: Long = keys.length

  def init(model: GNNPSModel, numBatch: Int): Int = {
    // init adjacent table on servers
    model.initNeighbors(keys, indptr, neighbours, numBatch)
    0
  }

  def toSemiGCNPartition(model: GNNPSModel, torchModelPath: String, useSecondOrder: Boolean,
                         testRatio: Float, numLabels: Int): GCNPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = if (numLabels > 1) {
      splitTrainTestM(model, testRatio)
    } else {
      if (model.nnzTestLabels() == 0) {
        val temp = splitTrainTest(model, testRatio)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      } else {
        val temp = getTrainTest(model)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      }
    }

    new GCNPartition(index, keys, indptr, neighbours,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath, useSecondOrder)
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

  def splitTrainTestM(model: GNNPSModel, testRatio: Float): (Array[Int], Array[Int], Array[Array[Float]], Array[Array[Float]]) = {
    val myLabels = model.readMultiLabels(keys)
    val it = myLabels.long2ObjectEntrySet().fastIterator()

    val trainIdx = new ArrayBuffer[Int]()
    val testIdx = new ArrayBuffer[Int]()
    val trainLabels = new ArrayBuffer[Array[Float]]()
    val testLabels = new ArrayBuffer[Array[Float]]()
    val position = new Long2IntOpenHashMap(keys.length)
    for (idx <- keys.indices)
      position.put(keys(idx), idx)

    while (it.hasNext) {
      val entry = it.next()
      val (node, labels_) = (entry.getLongKey, entry.getValue)
      if (labels_ != null && labels_.length > 0) {
        if (labels_(0) == 0) { // training node
          trainLabels.append(labels_.slice(1, labels_.length))
          trainIdx.append(position.get(node))
        } else { // testing node
          testLabels.append(labels_.slice(1, labels_.length))
          testIdx.append(position.get(node))
        }
      }
    }

    if (testIdx.isEmpty) {
      assert(testRatio > 0, s"testRatio is $testRatio while no testing samples.")
      val splitPoint = (trainIdx.length * (1 - testRatio)).toInt
      val (trainIdx_, testIdx_) = trainIdx.toArray.splitAt(splitPoint)
      val (trainLabels_, testLabels_) = trainLabels.toArray.splitAt(splitPoint)
      (trainIdx_, testIdx_, trainLabels_, testLabels_)
    } else {
      (trainIdx.toArray, testIdx.toArray, trainLabels.toArray, testLabels.toArray)
    }
  }

  def getTrainTest(model: GNNPSModel): (Array[Int], Array[Int], Array[Float], Array[Float]) = {
    val trainLabels = model.readLabels2(keys)
    val testLabels = model.readTestLabels(keys)
    val position = new Long2IntOpenHashMap(keys.length)
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

  def toMiniBatchDGIPartition(model: GNNPSModel, torchModelPath: String, useSecondOrder: Boolean, dataFormat: String): DGIPartition = {
    new DGIPartition(index, keys, indptr, neighbours, null, torchModelPath, useSecondOrder, dataFormat)
  }

  def toSemiGATPartition(model: GNNPSModel, torchModelPath: String, useSecondOrder: Boolean,
                         testRatio: Float, numLabels: Int): GATPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = if (numLabels > 1) {
      splitTrainTestM(model, testRatio)
    } else {
      if (model.nnzTestLabels() == 0) {
        val temp = splitTrainTest(model, testRatio)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      } else {
        val temp = getTrainTest(model)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      }
    }
    new GATPartition(index, keys, indptr, neighbours,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath, useSecondOrder)
  }

}

private[gcn]
class GraphAdjTypePartition(index: Int,
                            keys: Array[Long],
                            indptr: Array[Int],
                            neighbors: Array[Long],
                            types: Array[Int],
                            weights: Array[Float]=Array()) extends GraphAdjPartition(index, keys, indptr, neighbors) {
  override
  def init(model: GNNPSModel, numBatch: Int): Int = {
    model.initNeighbors(keys, indptr, neighbors, types, numBatch)
    0
  }

  def toSemiRGCNPartition(model: GNNPSModel, torchModelPath: String, useSecondOrder: Boolean,
                          testRatio: Float, numLabels: Int): RGCNPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = if (numLabels > 1) {
      splitTrainTestM(model, testRatio)
    } else {
      if (model.nnzTestLabels() == 0) {
        val temp = splitTrainTest(model, testRatio)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      } else {
        val temp = getTrainTest(model)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      }
    }
    new RGCNPartition(index, keys, indptr, neighbors, types,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath, useSecondOrder)
  }

  def toSemiHANPartition(model: GNNPSModel, torchModelPath: String, useSecondOrder: Boolean,
                         testRatio: Float, itemTypes: Int, numLabels: Int): HANPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = if (numLabels > 1) {
      splitTrainTestM(model, testRatio)
    } else {
      if (model.nnzTestLabels() == 0) {
        val temp = splitTrainTest(model, testRatio)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      } else {
        val temp = getTrainTest(model)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      }
    }
    new HANPartition(index, keys, indptr, neighbors, types,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath, itemTypes, useSecondOrder)
  }

  def toSemiHAggregatorPartition(model: GNNPSModel, torchModelPath: String, useSecondOrder: Boolean,
                                 testRatio: Float, numLabels: Int, isTraining: Boolean = true): HAggregatorPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = if (isTraining) {
      if (numLabels > 1) {
        splitTrainTestM(model, testRatio)
      } else {
        if (model.nnzTestLabels() == 0) {
          val temp = splitTrainTest(model, testRatio)
          (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
        } else {
          val temp = getTrainTest(model)
          (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
        }
      }
    } else {
      (null, null, null, null)
    }
    new HAggregatorPartition(index, keys, indptr, neighbors, types, weights,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath, useSecondOrder)
  }
}

private[gcn]
class GraphAdjWeightedPartition(index: Int,
                                keys: Array[Long],
                                indptr: Array[Int],
                                neighbors: Array[Long],
                                weights: Array[Float]) extends GraphAdjPartition(index, keys, indptr, neighbors) {
  override
  def init(model: GNNPSModel, numBatch: Int): Int = {
    model.initNeighbors(keys, indptr, neighbors, null, numBatch)
    0
  }

  override
  def toMiniBatchDGIPartition(model: GNNPSModel, torchModelPath: String, useSecondOrder: Boolean, dataFormat: String): DGIPartition = {
    new DGIPartition(index, keys, indptr, neighbors, weights, torchModelPath, useSecondOrder, dataFormat)
  }
}

private[gcn]
class GraphAdjBiPartition(index: Int,
                          keys: Array[Long],
                          indptr: Array[Int],
                          neighbors: Array[Long],
                          edgeTypes: Array[Int],
                          dstTypes: Array[Int],
                          graphType: Int,
                          hasEdgeType: Boolean = false,
                          hasNodeType: Boolean = false) extends GraphAdjPartition(index, keys, indptr, neighbors){

  def init(model: BiSAGEPSModel, numBatch: Int): Int = {
    if (hasEdgeType && hasNodeType) {
      model.initNeighbors(keys, indptr, neighbors, edgeTypes, dstTypes, graphType, numBatch)
    } else if (hasEdgeType) {
      model.initNeighbors(keys, indptr, neighbors, edgeTypes, graphType, numBatch, false)
    } else if (hasNodeType) {
      model.initNeighbors(keys, indptr, neighbors, dstTypes, graphType, numBatch, true)
    } else {
      model.initNeighbors(keys, indptr, neighbors, graphType, numBatch)
    }
    0
  }

  /** for IGMC algo
    * split edge into train dataset and valid dataset
    * where edge type == -1 means this edge has not type
    */
  def splitEdgeTrainTest(model: GNNPSModel, testRatio: Float):
  (Array[(Int, Int)], Array[(Int, Int)], Array[Int], Array[Int], Array[Long]) = {
    val labeledEdges = keys.zipWithIndex.flatMap{ pair =>
      (indptr(pair._2) until indptr(pair._2 + 1)).map(i => (pair._1, neighbors(i), edgeTypes(i)))
    }.filter(pair => pair._3 != -1)

    val size = labeledEdges.length
    val idxHasLabels = new Array[(Int, Int)](size)
    val labels = new Array[Int](size)
    val positionU = new Long2IntOpenHashMap(keys.size)
    val positionI = new Long2IntOpenHashMap()
    val keysI = new LongArrayList()
    for (idx <- keys.indices)
      positionU.put(keys(idx), idx)
    for (idx <- neighbors.indices) {
      if (!positionI.containsKey(neighbors(idx))) {
        keysI.add(neighbors(idx))
        positionI.put(neighbors(idx), positionI.size())
      }
    }

    var id = 0
    for (pair <- labeledEdges) {
      idxHasLabels(id) = (positionU.get(pair._1), positionI.get(pair._2))
      labels(id) = pair._3
      id += 1
    }

    val splitPoint = (size * (1 - testRatio)).toInt
    val randomIdx = randomIndex(size)
    val (trainSplitPoints, testSplitPoints) = randomIdx.splitAt(splitPoint)
    val trainIdx = trainSplitPoints.map(id => idxHasLabels(id))
    val testIdx = testSplitPoints.map(id => idxHasLabels(id))
    val trainLabels = trainSplitPoints.map(id => labels(id))
    val testLabels = testSplitPoints.map(id => labels(id))

    (trainIdx, testIdx, trainLabels, testLabels, keysI.toLongArray)
  }

  def randomIndex(maxIndex: Int): Array[Int] = {
    val seq = (0 until maxIndex).map(x=>x)
    Random.shuffle(seq).toArray
  }

  def toSemiBiGCNPartition(model: BiSAGEPSModel, torchModelPath: String, useSecondOrder: Boolean,
                           testRatio: Float, numLabels: Int): BiGCNPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = if (numLabels > 1) {
      splitTrainTestM(model, testRatio)
    } else {
      if (model.nnzTestLabels() == 0) {
        val temp = splitTrainTest(model, testRatio)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      } else {
        val temp = getTrainTest(model)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      }
    }
    new BiGCNPartition(index, keys, indptr, neighbors, edgeTypes, dstTypes,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath, useSecondOrder, hasEdgeType, hasNodeType)
  }

  def toSemiBiGCNPartition(model: BiSAGEPSModel, torchModelPath: String, useSecondOrder: Boolean): BiGCNPartition = {
    new BiGCNPartition(index, keys, indptr, neighbors, edgeTypes, dstTypes,
      null, null, null, null, torchModelPath,
      useSecondOrder, hasEdgeType, hasNodeType)
  }

  def toBiSAGEPartition(model: BiSAGEPSModel, torchModelPath: String, useSecondOrder: Boolean, dataFormat: String): BiSAGEPartition = {
    new BiSAGEPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder, dataFormat)
  }

  def toHGATPartition(model: BiSAGEPSModel, torchModelPath: String, useSecondOrder: Boolean, dataFormat: String): HGATPartition = {
    new HGATPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder, dataFormat)
  }

  def toIGMCPartition(model: BiSAGEPSModel, torchModelPath: String, useSecondOrder: Boolean,
                      testRatio: Float): IGMCPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels, keysI) = splitEdgeTrainTest(model, testRatio)
    new IGMCPartition(index, keys, indptr, neighbors, edgeTypes,
      trainIdx, trainLabels, testIdx, testLabels, keysI, torchModelPath, useSecondOrder, hasEdgeType)
  }
}

private[gcn]
class GraphAdjEdgePartition(index: Int,
                            keys: Array[Long],
                            indptr: Array[Int],
                            neighbors: Array[Long],
                            edgeDim: Int,
                            features: Array[IntFloatVector]) extends GraphAdjPartition(index, keys, indptr, neighbors) {
  override
  def init(model: GNNPSModel, numBatch: Int): Int = {
    model.initEdgeFeatures(keys, indptr, neighbors, features, numBatch)
    0
  }

  override
  def toSemiGCNPartition(model: GNNPSModel, torchModelPath: String, useSecondOrder: Boolean,
                         testRatio: Float, numLabels: Int): EdgePropGCNPartition = {
    val (trainIdx, testIdx, trainLabels, testLabels) = if (numLabels > 1) {
      splitTrainTestM(model, testRatio)
    } else {
      if (model.nnzTestLabels() == 0) {
        val temp = splitTrainTest(model, testRatio)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      } else {
        val temp = getTrainTest(model)
        (temp._1, temp._2, temp._3.map(Array(_)), temp._4.map(Array(_)))
      }
    }
    new EdgePropGCNPartition(index, keys, indptr, neighbors, edgeDim, features,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath, useSecondOrder)
  }
}


private[gcn]
class GraphAdjWeightedTypePartition(index: Int,
                            keys: Array[Long],
                            indptr: Array[Int],
                            neighbors: Array[Long],
                            types: Array[Int],
                            weights: Array[Float]) extends GraphAdjTypePartition(index, keys, indptr, neighbors, types) {
  override
  def init(model: GNNPSModel, numBatch: Int): Int = {
    model.initNeighbors(keys, indptr, neighbors, types, numBatch)
    0
  }
}

private[gcn]
class NodeFeaturePartition(index: Int,
                           keys: Array[Long],
                           features: Array[IntFloatVector]) extends Serializable {
  def init(model: GNNPSModel, numBatch: Int): Unit =
    model.initNodeFeatures(keys, features, numBatch)

  def init(model: BiSAGEPSModel, graphType: Int, numBatch: Int): Unit =
    model.initNodeFeatures(keys, features, graphType, numBatch)
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

object GraphAdjWeightedPartition {
  def apply(index: Int, iterator: Iterator[(Long, Iterable[(Long, Float)])]): GraphAdjWeightedPartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbors = new LongArrayList()
    val weights = new FloatArrayList()

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      for ((n, t) <- ns) {
        neighbors.add(n)
        weights.add(t)
      }
      indptr.add(neighbors.size())
      keys.add(node)
    }

    new GraphAdjWeightedPartition(index, keys.toLongArray,
      indptr.toIntArray,
      neighbors.toLongArray,
      weights.toFloatArray)
  }
}

object GraphAdjWeightedTypePartition {
  def apply(index: Int, iterator: Iterator[(Long, Iterable[(Long, Float, Int)])], hasWeighted: Boolean=false, hasUseWeightedAggregate: Boolean=false): GraphAdjWeightedTypePartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbors = new LongArrayList()
    val types = new IntArrayList()
    val weights_ = new FloatArrayList()

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      for ((n, w, t) <- ns) {
        neighbors.add(n)
        weights_.add(w)
        types.add(t)
      }
      indptr.add(neighbors.size())
      keys.add(node)
    }

    new GraphAdjWeightedTypePartition(index, keys.toLongArray,
      indptr.toIntArray,
      neighbors.toLongArray,
      types.toIntArray, weights_.toFloatArray)
  }
}

object GraphAdjBiPartition{
  def apply(index: Int, iterator: Iterator[(Long, Iterable[(Long, Int, Int)])], graphType: Int,
            hasEdgeType: Boolean = false, hasNodeType: Boolean = false): GraphAdjBiPartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbors = new LongArrayList()
    val edgeTypes = new IntArrayList()
    val dstTypes = new IntArrayList()

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      for ((n, t1, t2) <- ns) {
        neighbors.add(n)
        if (hasEdgeType) edgeTypes.add(t1)
        if (hasNodeType) dstTypes.add(t2)
      }
      indptr.add(neighbors.size())
      keys.add(node)
    }

    new GraphAdjBiPartition(index, keys.toLongArray,
      indptr.toIntArray,
      neighbors.toLongArray,
      edgeTypes.toIntArray,
      dstTypes.toIntArray,
      graphType,
      hasEdgeType,
      hasNodeType)
  }
}

private[gcn]
object GraphAdjEdgePartition {
  def apply(index: Int, iterator: Iterator[(Long, Iterable[(Long, Array[Float])])]): GraphAdjEdgePartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbours = new LongArrayList()
    val edges = new ArrayBuffer[IntFloatVector]()
    var edgeDim = 0

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      ns.foreach { n =>
        if (edgeDim == 0) {
          edgeDim = n._2.length
        }
        neighbours.add(n._1)
        edges.append(new IntFloatVector(n._2.length, new IntFloatDenseVectorStorage(n._2)))
      }
      indptr.add(neighbours.size())
      keys.add(node)
    }

    new GraphAdjEdgePartition(index, keys.toLongArray,
      indptr.toIntArray, neighbours.toLongArray, edgeDim, edges.toArray)
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