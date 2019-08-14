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

  def toSemiGCNPartition(model: GNNPSModel, torchModelPath: String, testRatio: Float): GCNPartition = {
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
    new GCNPartition(index, keys, indptr, neighbours,
      trainIdx, trainLabels, testIdx, testLabels, torchModelPath)
  }

  def toMiniBatchDGIPartition(model:GNNPSModel, torchModelPath: String, useSecondOrder: Boolean): DGIPartition = {
    new DGIPartition(index, keys, indptr, neighbours, torchModelPath, useSecondOrder)
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


