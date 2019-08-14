package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

class GNNPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   torchModelPath: String,
                   useSecondOrder: Boolean) extends Serializable {

  def numNodes: Long = keys.length

  def numEdges: Long = neighbors.length

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: GNNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int): (Double, Long) = ???

  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int): Long = ???

  def genLabels(batchSize: Int,
                model: GNNPSModel,
                featureDim: Int,
                numSample: Int): Iterator[(Long, Long, String)] = ???


  def genEmbedding(batchSize: Int,
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int): Iterator[(Long, String)] = {


    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val batchKeys = new LongOpenHashSet()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()

    val it = new Iterator[Array[(Long, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext)
          TorchModel.addModel(torch)
        return batchIterator.hasNext
      }

      override def next: Array[(Long, String)] = {
        val batch = batchIterator.next().toArray
        val output = genEmbeddingBatch(batch, model, featureDim, numSample,
          srcs, dsts, batchKeys, index, weights, torch)
        srcs.clear()
        dsts.clear()
        batchKeys.clear()
        index.clear()
        output.toArray
      }
    }
    it.flatMap(f => f.iterator)
  }


  def genEmbeddingBatch(batchIdx: Array[Int],
                        model: GNNPSModel,
                        featureDim: Int,
                        numSample: Int,
                        srcs: LongArrayList,
                        dsts: LongArrayList,
                        batchKeys: LongOpenHashSet,
                        index: Long2IntOpenHashMap,
                        weights: Array[Float],
                        torch: TorchModel): Iterator[(Long, String)] = {
    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    val batchIds = batchIdx.map(idx => keys(idx))
    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx, keys, indptr, neighbors,
      srcs, dsts, batchKeys, index, numSample, model, true)
    val x = MakeFeature.makeFeatures(index, featureDim, model)
    val output = torch.gcnEmbedding(batchIdx.length, x, featureDim,
      first, second, weights)
    assert(output.length % batchIdx.length == 0)
    val outputDim = output.length / batchIdx.length
    output.sliding(outputDim, outputDim)
      .zip(batchIds.iterator).map {
      case (h, key) => // h is representation for node ``key``
        (key, h.mkString(","))
    }
  }

}
