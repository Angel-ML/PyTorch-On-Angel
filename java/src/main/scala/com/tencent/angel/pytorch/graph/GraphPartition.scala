package com.tencent.angel.pytorch.graph

import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.storage.{IntFloatDenseVectorStorage, IntFloatSortedVectorStorage, IntFloatSparseVectorStorage}
import com.tencent.angel.ml.math2.vector.{IntFloatVector, LongFloatVector}
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.{Long2FloatOpenHashMap, LongArrayList}

import scala.collection.mutable.ArrayBuffer

private[graph]
class GraphAdjPartition(index: Int,
                        keys: Array[Long],
                        indptr: Array[Int],
                        neighbours: Array[Long]) extends Serializable {

  def init(model: GraphPSModel): Int = {
    // init adjacent table on servers
    model.initNeighbours(keys, indptr, neighbours)
    0
  }

  def toGCNPartition(model: GraphPSModel, torchModelPath: String): GraphGCNPartition = {
    val labels = model.readLabels(keys).get(keys)
    new GraphGCNPartition(index, keys, indptr, neighbours, labels, torchModelPath)
  }

  def toMiniBatchGCNPartition(model: GraphPSModel, torchModelPath: String): GraphMiniBatchGCNPartition = {
    val labels = model.readLabels(keys).get(keys)
    new GraphMiniBatchGCNPartition(index, keys, indptr, neighbours, labels, torchModelPath)
  }
}

private[graph]
class NodeFeaturePartition(index: Int,
                           keys: Array[Long],
                           features: Array[IntFloatVector]) extends Serializable {
  def init(model: GraphPSModel): Unit =
    model.initNodeFeatures(keys, features)
}

private[graph]
class NodeLabelPartition(index: Int, labels: LongFloatVector) extends Serializable {
  def init(model: GraphPSModel): Unit =
    model.setLabels(labels)
}

private[graph]
class GraphGCNPartition(index: Int,
                        keys: Array[Long],
                        indptr: Array[Int],
                        neighbours: Array[Long],
                        labels: Array[Float],
                        torchModelPath: String) extends Serializable {

  def check(model: GraphPSModel): Unit = {
    val features = model.getFeatures(keys)
    val it = features.long2ObjectEntrySet().fastIterator()
    while (it.hasNext) {
      val entry = it.next()
      val (key, f) = (entry.getLongKey, entry.getValue)
      f.getStorage match {
        case sorted: IntFloatSortedVectorStorage =>
          println(s"${key}: sorted ${sorted.getIndices.zip(sorted.getValues).map(f => s"${f._1}:${f._2}").mkString(",")}")
        case sparse: IntFloatSparseVectorStorage =>
          println(s"${key}: sparse ")
        case dense : IntFloatDenseVectorStorage =>
          println(s"${key}: dense")
      }
    }
  }

  /* full batch functions */
  def train(curEpoch: Int, model: GraphPSModel, featureDim: Int, optim: AsyncOptim): Double = {
    val edgeIndex = makeEdgeIndex(keys, model)
    val x = makeFeatures(keys, model, featureDim)
    val weights = model.readWeights()
    val labels = makeLabels(keys.length)
    TorchModel.setPath(torchModelPath)
    val loss = TorchModel.get().gcnBackward(keys.length, x, featureDim, edgeIndex, weights, labels)
    model.step(weights, optim)
    loss
  }


  def predict(model: GraphPSModel, featureDim: Int): (Int, Int) = {
    val edgeIndex = makeEdgeIndex(keys, model)
    val x = makeFeatures(keys, model, featureDim)
    val weights = model.readWeights()
    val labels = makeLabels(keys.length)
    TorchModel.setPath(torchModelPath)
    val output = TorchModel.get().gcnForward(keys.length, x, featureDim, edgeIndex, weights)
    assert(labels.length == output.length)
    var right = 0
    for (i <- labels.indices)
      if (labels(i) == output(i))
        right += 1
    return (right, labels.length)
  }

  def makeEdgeIndex(nodes: Array[Long], model: GraphPSModel): Array[Long] = {
    val neighbors = model.sampleNeighbors(nodes, -1)
    var size: Int = 0
    var it = neighbors.long2ObjectEntrySet().fastIterator()
    while (it.hasNext)
      size += it.next().getValue.length

    val edgeIndex = new Array[Long](size * 2)
    it = neighbors.long2ObjectEntrySet().fastIterator()
    var idx = 0
    while (it.hasNext) {
      val entry = it.next()
      val (src, dsts) = (entry.getLongKey, entry.getValue)
      var j = 0
      while (j < dsts.length) {
        edgeIndex(idx) = src
        edgeIndex(idx + size) = dsts(j)
        idx += 1
        j += 1
      }
    }

    edgeIndex
  }

  def makeFeatures(nodes: Array[Long], model: GraphPSModel, featureDim: Int): Array[Float] = {
    val features = model.getFeatures(nodes)
    val x = new Array[Float](nodes.length * featureDim)
    val it = features.long2ObjectEntrySet().fastIterator()
    while (it.hasNext) {
      val entry = it.next()
      val (node, f) = (entry.getLongKey, entry.getValue)
      makeFeature(node.toInt * featureDim, f, x)
    }
    x
  }

  def makeFeature(start: Int, f: IntFloatVector, x: Array[Float]): Unit = {
    f.getStorage match {
      case sorted: IntFloatSortedVectorStorage =>
        val indices = sorted.getIndices
        val values = sorted.getValues
        var j = 0
        while (j < indices.length) {
          x(start + indices(j)) = values(j)
          j += 1
        }
    }
  }

  def makeLabels(nodes: Array[Long]): Array[Float] = {
    val map = new Long2FloatOpenHashMap(keys.length)
    for (idx <- keys.indices)
      map.put(keys(idx), labels(idx))

    val results = new Array[Float](nodes.length)
    for (idx <- nodes.indices)
      results(idx) = map.get(nodes(idx))

    results
  }

  def makeLabels(maxId: Int): Array[Long] = {
    val map = new Long2FloatOpenHashMap(keys.length)
    for (idx <- keys.indices)
      map.put(keys(idx), labels(idx))

    val results = new Array[Long](maxId)
    for (k <- 0 until maxId)
      results(k) = map.get(k).toLong
    results
  }
}


private[graph]
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

private[graph]
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

private[graph]
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


