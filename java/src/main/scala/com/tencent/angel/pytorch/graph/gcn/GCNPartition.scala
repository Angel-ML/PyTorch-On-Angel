package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.ints.{IntArrayList, IntOpenHashSet}
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

import scala.collection.mutable.ArrayBuffer

class GCNPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   trainIdx: Array[Int],
                   trainLabels: Array[Float],
                   testIdx: Array[Int],
                   testLabels: Array[Float],
                   torchModelPath: String) extends
  GNNPartition(index, keys, indptr, neighbors, torchModelPath, true) {

  def getTrainTestSize(): (Int, Int) = (trainIdx.length, testIdx.length)

  override
  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int): (Double, Long) = {
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val batchKeys = new LongOpenHashSet()
    val batchIterator = trainIdx.zip(trainLabels).sliding(batchSize, batchSize)
    var lossSum = 0.0
    var numRight: Long = 0

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()

    while (batchIterator.hasNext) {
      val batch = batchIterator.next()
      val (loss, right) = trainBatch(batch, model, featureDim,
        optim, numSample, srcs, dsts, batchKeys, index, torch)
      lossSum += loss * batch.length
      numRight += right
      srcs.clear()
      dsts.clear()
      batchKeys.clear()
      index.clear()
    }

    TorchModel.addModel(torch) // return torch for next epoch
    (lossSum, numRight)
  }

  def trainBatch(batchIdx: Array[(Int, Float)],
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 srcs: LongArrayList,
                 dsts: LongArrayList,
                 batchKeys: LongOpenHashSet,
                 index: Long2IntOpenHashMap,
                 torch: TorchModel): (Double, Long) = {
    val targets = new Array[Long](batchIdx.length)
    var k = 0
    for ((idx, label) <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
      targets(k) = label.toLong
      k += 1
    }

    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx.map(f => f._1),
      keys, indptr, neighbors, srcs, dsts,
      batchKeys, index, numSample, model, true)
    val x = MakeFeature.makeFeatures(index, featureDim, model)

    val weights = model.readWeights()
    val outputs = torch.gcnPredict(batchIdx.length, x, featureDim,
      first, second, weights)
    val loss = torch.gcnBackward(batchIdx.length, x, featureDim,
      first, second, weights, targets)
    model.step(weights, optim)

    var right: Long = 0
    for (i <- outputs.indices)
      if (outputs(i) == targets(i))
        right += 1
    (loss, right)
  }

  override
  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int): Long = {
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val batchKeys = new LongOpenHashSet()
    val batchIterator = testIdx.zip(testLabels).sliding(batchSize, batchSize)
    var numRight: Int = 0
    val weights = model.readWeights()

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()

    while (batchIterator.hasNext) {
      val batch = batchIterator.next()
      val right = predictBatch(batch, model, featureDim, numSample,
        srcs, dsts, batchKeys, index, weights, torch)
      numRight += right
      srcs.clear()
      dsts.clear()
      batchKeys.clear()
      index.clear()
    }

    TorchModel.addModel(torch) // return torch for next epoch
    numRight
  }

  def predictBatch(batchIdx: Array[(Int, Float)],
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   srcs: LongArrayList,
                   dsts: LongArrayList,
                   batchKeys: LongOpenHashSet,
                   index: Long2IntOpenHashMap,
                   weights: Array[Float],
                   torch: TorchModel): Int = {
    val targets = new Array[Long](batchIdx.length)
    var k = 0
    for ((idx, label) <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
      targets(k) = label.toLong
      k += 1
    }

    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx.map(f => f._1),
      keys, indptr, neighbors, srcs, dsts,
      batchKeys, index, numSample, model, true)
    val x = MakeFeature.makeFeatures(index, featureDim, model)
    val outputs = torch.gcnPredict(batchIdx.length, x, featureDim,
      first, second, weights)
    assert(outputs.length == targets.length)
    var right = 0
    for (i <- outputs.indices)
      if (outputs(i) == targets(i))
        right += 1
    right
  }

  override
  def genLabels(batchSize: Int,
                model: GNNPSModel,
                featureDim: Int,
                numSample: Int): Iterator[(Long, Long, String)] = {
//    val labeledIdx = new IntOpenHashSet()
//    trainIdx.foreach(idx => labeledIdx.add(idx))
//    testIdx.foreach(idx => labeledIdx.add(idx))
//
//    val unlabeledIdx = new IntArrayList()
//    for (idx <- keys.indices)
//      if (!labeledIdx.contains(idx))
//        unlabeledIdx.add(idx)
//
//    val predictIdx = unlabeledIdx.toIntArray

    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val batchKeys = new LongOpenHashSet()
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)

    val it = new Iterator[Array[(Long, Long, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.addModel(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, Long, String)] = {
        val batch = batchIterator.next().toArray
        val outputs = genLabelsBatch(batch, model, featureDim, numSample,
          srcs, dsts, batchKeys, index, weights, torch)
        srcs.clear()
        dsts.clear()
        batchKeys.clear()
        index.clear()
        outputs.toArray
      }
    }

    it.flatMap(f => f.iterator)
  }

  def genLabelsBatch(batchIdx: Array[Int],
                     model: GNNPSModel,
                     featureDim: Int,
                     numSample: Int,
                     srcs: LongArrayList,
                     dsts: LongArrayList,
                     batchKeys: LongOpenHashSet,
                     index: Long2IntOpenHashMap,
                     weights: Array[Float],
                     torch: TorchModel): Iterator[(Long, Long, String)] = {
    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    var (start, end) = (0L, 0L)

    start = System.currentTimeMillis()
    val batchIds = batchIdx.map(idx => keys(idx))
    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx,
      keys, indptr, neighbors,
      srcs, dsts, batchKeys, index, numSample, model, true)
    val x = MakeFeature.makeFeatures(index, featureDim, model)
    end = System.currentTimeMillis()
    val networkTime = end - start
    val outputs = torch.gcnForward(batchIds.length, x, featureDim,
      first, second, weights)
    end = System.currentTimeMillis()
    val forwardTime = System.currentTimeMillis() - end
    println(s"networkTime=$networkTime forwardTime=$forwardTime")
    assert(outputs.length % batchIds.length == 0)
    val numLabels = outputs.length / batchIds.length
    outputs.sliding(numLabels, numLabels)
      .zip(batchIds.iterator).map {
      case (p, key) =>
        val maxIndex = p.zipWithIndex.maxBy(_._1)._2
        (key, maxIndex, p.mkString(","))
    }
  }

}
