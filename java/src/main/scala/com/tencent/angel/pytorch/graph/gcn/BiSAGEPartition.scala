package com.tencent.angel.pytorch.graph.gcn

import java.util.{HashMap => JHashMap, Map => JMap}
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList}

private[gcn]
class BiSAGEPartition(index: Int,
                      keys: Array[Long],
                      indptr: Array[Int],
                      neighbors: Array[Long],
                      torchModelPath: String,
                      useSecondOrder: Boolean,
                      dataFormat: String) extends
  GNNPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder) {

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: BiSAGEPSModel,
                 userFeatureDim: Int,
                 itemFeatureDim: Int,
                 optim: AsyncOptim,
                 userNumSample: Int,
                 itemNumSample: Int,
                 graphType: Int,
                 userFieldNum: Int,
                 itemFieldNum: Int,
                 fieldMultiHot: Boolean,
                 trainRatio: Float): (Double, Long, Int) = {
    val batchIterator = sampleTrainData(keys.indices, trainRatio).sliding(batchSize, batchSize)
    var lossSum = 0.0
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0
    while (batchIterator.hasNext) {
      val batch = batchIterator.next().toArray
      val loss = trainBatch(batch, model, userFeatureDim, itemFeatureDim, optim, userNumSample, itemNumSample, torch, graphType, userFieldNum, itemFieldNum, fieldMultiHot)
      lossSum += loss * batch.length
      numStep += 1
    }

    TorchModel.put(torch)
    (lossSum, keys.length, numStep)
  }

  def trainBatch(batchIdx: Array[Int],
                 model: BiSAGEPSModel,
                 userFeatureDim: Int,
                 itemFeatureDim: Int,
                 optim: AsyncOptim,
                 userNumSample: Int,
                 itemNumSample: Int,
                 torch: TorchModel,
                 graphType: Int,
                 userFieldNum: Int,
                 itemFieldNum: Int,
                 fieldMultiHot: Boolean): Double = {
    val params = makeParams(batchIdx, userNumSample, itemNumSample, userFeatureDim, itemFeatureDim, model, graphType, true, userFieldNum, itemFieldNum, fieldMultiHot)
    val weights = model.readWeights()
    params.put("weights", weights)
    val loss = torch.gcnBackward(params, userFieldNum > 0)
    model.step(weights, optim)

    if (userFieldNum > 0) {
      //check if the grad really replaced the pulledUEmbedding
      val pulledUEmbedding = params.get("pulledUEmbedding").asInstanceOf[Array[Vector]]
      val uFeats = params.get("uFeats").asInstanceOf[Array[Int]]
      val uEmbeddingInput = params.get("pos_u").asInstanceOf[Array[Float]]
      val embeddingDim = pulledUEmbedding.length
      makeEmbeddingGrad(uEmbeddingInput, pulledUEmbedding, uFeats, embeddingDim)
      model.asInstanceOf[SparseBiSAGEPSModel].updateUserEmbedding(pulledUEmbedding, optim)
      if (itemFeatureDim > 0) {
        val pulledIEmbedding = params.get("pulledIEmbedding").asInstanceOf[Array[Vector]]
        val iFeats = params.get("iFeats").asInstanceOf[Array[Int]]
        val iEmbeddingInput = params.get("pos_i").asInstanceOf[Array[Float]]
        val embeddingDim = pulledIEmbedding.length
        makeEmbeddingGrad(iEmbeddingInput, pulledIEmbedding, iFeats, embeddingDim)
        model.asInstanceOf[SparseBiSAGEPSModel].updateItemEmbedding(pulledIEmbedding, optim)
      }
    }

    loss
  }

  def makeParams(batchIdx: Array[Int],
                 userNumSample: Int,
                 itemNumSample: Int,
                 userFeatureDim: Int,
                 itemFeatureDim: Int,
                 model: BiSAGEPSModel,
                 graphType: Int,
                 isTraining: Boolean,
                 userFieldNum: Int,
                 itemFieldNum: Int,
                 fieldMultiHot: Boolean): JMap[String, Object] = {
    val srcIndex = new Long2IntOpenHashMap()
    val dstIndex = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val srcs_ = new LongArrayList()
    val dsts_ = new LongArrayList()
    for (idx <- batchIdx)
      srcIndex.put(keys(idx), srcIndex.size())

    val emptyEdge = new Array[Long](1*2)
    emptyEdge(0) = -1
    emptyEdge(1) = -1
    // sample edges u-i-u-i, i-u-i-u
    val sampleNums_first = if (graphType==0) userNumSample else itemNumSample
    val sampleNums_second = if (graphType==0) itemNumSample else userNumSample

    val (first, firstKeys) = MakeBiEdgeIndex.makeFirstOrderEdgeIndex(
      batchIdx, keys, indptr, neighbors, srcs, dsts, srcIndex, dstIndex, sampleNums_first)
    val (second, secondKeys) = MakeBiEdgeIndex.makeSecondOrderEdgeIndex(firstKeys, srcs_, dsts_,
      dstIndex, srcIndex, model, sampleNums_second, 1-graphType)

    val (second_, _) = if(!useSecondOrder) (emptyEdge, null)
    else {
      MakeBiEdgeIndex.makeSecondOrderEdgeIndex(secondKeys, srcs, dsts, srcIndex, dstIndex, model, sampleNums_first, graphType)
    }

    val params = new JHashMap[String, Object]()

    val (pos_u, uBatchIds, uFieldIds) = if (graphType == 0)
      MakeSparseBiFeature.makeFeatures(srcIndex, userFeatureDim, model, 0, params, userFieldNum, fieldMultiHot)
    else
      MakeSparseBiFeature.makeFeatures(dstIndex, userFeatureDim, model, 0, params, userFieldNum, fieldMultiHot)

    val (pos_i, iBatchIds, iFieldIds) = if (itemFeatureDim > 0) {
      if (graphType == 0)
        MakeSparseBiFeature.makeFeatures(dstIndex, itemFeatureDim, model, 1, params, itemFieldNum, fieldMultiHot)
      else
        MakeSparseBiFeature.makeFeatures(srcIndex, itemFeatureDim, model, 1, params, itemFieldNum, fieldMultiHot)
    } else (null, null, null)

    val (neg_u, negUBatchIds, negUFieldIds) = if (isTraining && graphType == 0)
      MakeSparseBiFeature.sampleFeatures(srcIndex.size(), userFeatureDim, model, 0, dataFormat, params, userFieldNum, fieldMultiHot)
    else if (isTraining && graphType == 1)
      MakeSparseBiFeature.sampleFeatures(dstIndex.size(), userFeatureDim, model, 0, dataFormat, params, userFieldNum, fieldMultiHot)
    else (null, null, null)

    val (neg_i, negIBatchIds, negIFieldIds) = if (isTraining && itemFeatureDim > 0) {
      if (isTraining && graphType == 0)
        MakeSparseBiFeature.sampleFeatures(dstIndex.size(), itemFeatureDim, model, 1, dataFormat, params, itemFieldNum, fieldMultiHot)
      else if (isTraining && graphType == 1)
        MakeSparseBiFeature.sampleFeatures(srcIndex.size(), itemFeatureDim, model, 1, dataFormat, params, itemFieldNum, fieldMultiHot)
      else (null, null, null)
    } else (null, null, null)

    if (isTraining && !fieldMultiHot)
      assert(pos_u.length == neg_u.length)
    if(isTraining && itemFeatureDim > 0 && !fieldMultiHot) {
      assert(pos_i.length == neg_i.length)
    }


    if(graphType == 0){
      // sample u-i-u-i
      params.put("first_u_edge_index", first)
      params.put("first_i_edge_index", second)
      params.put("second_u_edge_index", second_)
      params.put("second_i_edge_index", emptyEdge)
    }
    else{
      // sample i-u-i-u
      params.put("first_u_edge_index", second)
      params.put("first_i_edge_index", first)
      params.put("second_u_edge_index", emptyEdge)
      params.put("second_i_edge_index", second_)
    }
    if (isTraining) {
      params.put("pos_u", pos_u)
      params.put("neg_u", neg_u)
      if (userFieldNum > 0) {
        params.put("u_batch_ids", uBatchIds)
        params.put("u_field_ids", uFieldIds)
        params.put("neg_u_batch_ids", negUBatchIds)
        params.put("neg_u_field_ids", negUFieldIds)
      }
      if (itemFeatureDim > 0){
        params.put("pos_i", pos_i)
        params.put("neg_i", neg_i)
        if (itemFieldNum > 0) {
          params.put("i_batch_ids", iBatchIds)
          params.put("i_field_ids", iFieldIds)
          params.put("neg_i_batch_ids", negIBatchIds)
          params.put("neg_i_field_ids", negIFieldIds)
        }
      }
    }else{
      params.put("u", pos_u)
      if (itemFeatureDim > 0){
        params.put("i", pos_i)
      }

      if (userFieldNum > 0) {
        params.put("u_batch_ids", uBatchIds)
        params.put("u_field_ids", uFieldIds)
      }
      if (itemFeatureDim > 0){
        if (itemFieldNum > 0) {
          params.put("i_batch_ids", iBatchIds)
          params.put("i_field_ids", iFieldIds)
        }
      }
    }
    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(userFeatureDim))
    params.put("user_feature_dim", new Integer(userFeatureDim))
    if (itemFeatureDim > 0){
      params.put("item_feature_dim", new Integer(itemFeatureDim))
    }
    params
  }

  def genEmbedding(batchSize: Int,
                   model: BiSAGEPSModel,
                   userFeatureDim: Int,
                   itemFeatureDim: Int,
                   userNumSample: Int,
                   itemNumSample: Int,
                   numPartitions: Int,
                   graphType: Int,
                   userFieldNum: Int,
                   itemFieldNum: Int,
                   fieldMultiHot: Boolean): Iterator[(Long, String)] = {

    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()

    val keyIterator = new Iterator[Array[(Long, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, String)] = {
        val batch = batchIterator.next().toArray
        val output = genEmbeddingBatch(batch, model, userFeatureDim, itemFeatureDim,
          userNumSample, itemNumSample, weights, torch, graphType, userFieldNum, itemFieldNum, fieldMultiHot)
        output.toArray
      }
    }

    keyIterator.flatMap(f => f.iterator)
  }

  def genEmbeddingBatch(batchIdx: Array[Int],
                        model: BiSAGEPSModel,
                        userFeatureDim: Int,
                        itemFeatureDim: Int,
                        userNumSample: Int,
                        itemNumSample: Int,
                        weights: Array[Float],
                        torch: TorchModel,
                        graphType: Int,
                        userFieldNum: Int,
                        itemFieldNum: Int,
                        fieldMultiHot: Boolean): Iterator[(Long, String)] = {

    val batchIds = batchIdx.map(idx => keys(idx))
    val params = makeParams(batchIdx, userNumSample, itemNumSample, userFeatureDim, itemFeatureDim, model, graphType,
      false, userFieldNum, itemFieldNum, fieldMultiHot)
    params.put("weights", weights)
    val method = if(graphType == 0) "user_embedding_" else "item_embedding_"
    val output = torch.gcnBiEmbedding(params, method)
    assert(output.length % batchIdx.length == 0)
    val outputDim = output.length / batchIdx.length
    output.sliding(outputDim, outputDim)
      .zip(batchIds.iterator).map {
      case (h, key) => // h is representation for node ``key``
        (key, h.mkString(","))
    }
  }
}