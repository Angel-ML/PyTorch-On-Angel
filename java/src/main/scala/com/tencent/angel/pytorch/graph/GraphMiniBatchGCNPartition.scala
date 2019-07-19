package com.tencent.angel.pytorch.graph

import java.util.Random

import com.tencent.angel.ml.math2.storage.{IntFloatDenseVectorStorage, IntFloatSortedVectorStorage}
import com.tencent.angel.ml.math2.vector.IntFloatVector
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongArrays, LongOpenHashSet}


private[graph]
class GraphMiniBatchGCNPartition(index: Int,
                                 keys: Array[Long],
                                 indptr: Array[Int],
                                 neighbors: Array[Long],
                                 labels: Array[Float],
                                 torchModelPath: String) extends Serializable {
  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GraphPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int): (Double, Long) = {
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val batchKeys = new LongOpenHashSet()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    var lossSum = 0.0
    while (batchIterator.hasNext) {
      val batch = batchIterator.next().toArray
      val loss = trainBatch(batch, model, featureDim, optim, numSample, srcs, dsts, batchKeys, index)
      lossSum += loss * batch.length
      srcs.clear()
      dsts.clear()
      batchKeys.clear()
      index.clear()
    }
    (lossSum, keys.length)
  }

  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: GraphPSModel,
                   featureDim: Int,
                   numSample: Int): (Long, Long) = {
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val batchKeys = new LongOpenHashSet()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    var numRight: Int = 0
    while (batchIterator.hasNext) {
      val batch = batchIterator.next().toArray
      val right = predictBatch(batch, model, featureDim, numSample, srcs, dsts, batchKeys, index)
      numRight += right
      srcs.clear()
      dsts.clear()
      batchKeys.clear()
      index.clear()
    }
    (numRight, keys.length)
  }

  def trainBatch(batchIdx: Array[Int],
                 model: GraphPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 srcs: LongArrayList,
                 dsts: LongArrayList,
                 batchKeys: LongOpenHashSet,
                 index: Long2IntOpenHashMap): Double = {
    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    val (firstEdgeIndex, firstKeys) = makeFirstOrderEdgeIndex(batchIdx, srcs,
      dsts, batchKeys, index, numSample)
    val secondEdgeIndex = makeSecondOrderEdgeIndex(batchKeys, firstKeys, srcs,
      dsts, index, model, numSample)
    val x = makeFeatures(index, featureDim, model)
    val targets = batchIdx.map(idx => labels(idx).toLong)
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
//    println(s"first_edge_index.length=${firstEdgeIndex.length} second_edge_index.length=${secondEdgeIndex.length}")
//    println(s"first.row.max=${maxRow(firstEdgeIndex)}")
//    println(s"second.row.max=${maxRow(secondEdgeIndex)}")
//    println(s"index.size()=${index.size()} x.size(0)=${x.length / featureDim}")
    val loss = TorchModel.get().gcnBackward(batchIdx.length, x, featureDim,
      firstEdgeIndex, secondEdgeIndex, weights, targets)
    model.step(weights, optim)
    loss
  }

  def predictBatch(batchIdx: Array[Int],
                   model: GraphPSModel,
                   featureDim: Int,
                   numSample: Int,
                   srcs: LongArrayList,
                   dsts: LongArrayList,
                   batchKeys: LongOpenHashSet,
                   index: Long2IntOpenHashMap): Int = {
    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }
    val (firstEdgeIndex, firstKeys) = makeFirstOrderEdgeIndex(batchIdx, srcs,
      dsts, batchKeys, index, numSample)
    val secondEdgeIndex = makeSecondOrderEdgeIndex(batchKeys, firstKeys, srcs,
      dsts, index, model, numSample)
    val x = makeFeatures(index, featureDim, model)
    val targets = batchIdx.map(idx => labels(idx).toLong)
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val outputs = TorchModel.get().gcnForward(batchIdx.length, x, featureDim,
      firstEdgeIndex, secondEdgeIndex, weights)
    assert(outputs.length == targets.length)
    var right = 0
    for (i <- outputs.indices)
      if (outputs(i) == targets(i))
        right += 1
    right
  }

  def makeFirstOrderEdgeIndex(batchIdx: Array[Int],
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

  def makeSecondOrderEdgeIndex(batchKeys: LongOpenHashSet,
                               firstKeys: LongOpenHashSet,
                               srcs: LongArrayList,
                               dsts: LongArrayList,
                               index: Long2IntOpenHashMap,
                               model: GraphPSModel,
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

  def makeFeatures(index: Long2IntOpenHashMap, featureDim: Int, model: GraphPSModel): Array[Float] = {
    val size = index.size()
    val x = new Array[Float](size * featureDim)
    val keys = index.keySet().toLongArray
    val features = model.getFeatures(keys)
    val it = features.long2ObjectEntrySet().fastIterator()
    while (it.hasNext) {
      val entry = it.next()
      val (node, f) = (entry.getLongKey, entry.getValue)
      val start = index.get(node) * featureDim
      makeFeature(start, f, x)
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
      case dense: IntFloatDenseVectorStorage =>
        val values = dense.getValues
        var j = 0
        while (j < values.length) {
          x(start + j) = values(j)
          j += 1
        }
    }
  }

  def maxRow(edgeIndex: Array[Long]): Long = {
    var maxId: Long = -1
    val length = edgeIndex.length / 2
    for (i <- 0 until length)
      maxId = math.max(maxId, edgeIndex(i))
    maxId
  }
}
