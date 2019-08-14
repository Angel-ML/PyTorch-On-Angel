package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

import scala.collection.mutable.ArrayBuffer

private[gcn]
class DGIPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   torchModelPath: String,
                   useSecondOrder: Boolean) extends
  GNNPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder) {

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
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    var lossSum = 0.0
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    while (batchIterator.hasNext) {
      val batch = batchIterator.next().toArray
      val loss = trainBatch(batch, model, featureDim, optim, numSample,
        srcs, dsts, batchKeys, index, torch)
      lossSum += loss * batch.length
      srcs.clear()
      dsts.clear()
      batchKeys.clear()
      index.clear()
    }

    TorchModel.addModel(torch)
    (lossSum, keys.length)
  }

  def trainBatch(batchIdx: Array[Int],
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 srcs: LongArrayList,
                 dsts: LongArrayList,
                 batchKeys: LongOpenHashSet,
                 index: Long2IntOpenHashMap,
                 torch: TorchModel): Double = {

    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx,
      keys, indptr, neighbors, srcs, dsts, batchKeys,
      index, numSample, model, useSecondOrder)

    val posx = MakeFeature.makeFeatures(index, featureDim, model)
    val negx = MakeFeature.sampleFeatures(index.size(), featureDim, model)
    assert(posx.length == negx.length)
    val weights = model.readWeights()

    val loss = torch.dgiBackward(batchIdx.length, posx, negx,
      featureDim, first, second, weights)

    model.step(weights, optim)
    loss
  }

}
