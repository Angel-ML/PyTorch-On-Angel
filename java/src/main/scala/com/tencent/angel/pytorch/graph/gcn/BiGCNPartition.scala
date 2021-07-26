package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import java.util.{HashMap => JHashMap, Map => JMap}
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList}
import com.tencent.angel.ml.math2.vector.Vector

import scala.util.Random

class BiGCNPartition(index: Int,
                     keys: Array[Long],
                     indptr: Array[Int],
                     neighbors: Array[Long],
                     edgeTypes: Array[Int],
                     dstTypes: Array[Int],
                     userTrainIdx: Array[Int],
                     userTrainLabels: Array[Array[Float]],
                     userTestIdx: Array[Int],
                     userTestLabels: Array[Array[Float]],
                     torchModelPath: String,
                     useSecondOrder: Boolean,
                     hasEdgeType: Boolean,
                     hasNodeType: Boolean)
  extends GNNPartition(index, keys, indptr, neighbors, torchModelPath, useSecondOrder) {

  def getTrainTestSize(): (Int, Int) = (userTrainIdx.length, userTestIdx.length)

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: BiSAGEPSModel,
                 userFeatureDim: Int,
                 itemFeatureDim: Int,
                 optim: AsyncOptim,
                 userNumSample: Int,
                 itemNumSample: Int,
                 graphType: Int,
                 isSharedSamples: Boolean,
                 userFieldNum: Int,
                 itemFieldNum: Int,
                 fieldMultiHot: Boolean): (Double, Long, Int,  Array[(Float, Float)]) = {
    val batchIterator = userTrainIdx.zip(userTrainLabels).sliding(batchSize, batchSize)
    var lossSum = 0.0

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0

    val pairs = new Iterator[Array[(Float, Float)]] with Serializable {
      override def hasNext: Boolean = {
        batchIterator.hasNext
      }

      override def next: Array[(Float, Float)] = {
        val batch = batchIterator.next()
        val (loss, output) = trainBatch(batch, model, userFeatureDim, itemFeatureDim, optim, userNumSample, itemNumSample, torch, graphType, isSharedSamples, userFieldNum, itemFieldNum, fieldMultiHot)
        lossSum += loss * batch.length
        numStep += 1
        output._1.zip(output._2)
      }
    }.flatMap(p =>p).toArray

    TorchModel.put(torch)
    (lossSum, keys.length, numStep, pairs)
  }

  def trainBatch(batchIdx: Array[(Int, Array[Float])],
                 model: BiSAGEPSModel,
                 userFeatureDim: Int,
                 itemFeatureDim: Int,
                 optim: AsyncOptim,
                 userNumSample: Int,
                 itemNumSample: Int,
                 torch: TorchModel,
                 graphType: Int,
                 isSharedSamples: Boolean,
                 userFieldNum: Int,
                 itemFieldNum: Int,
                 fieldMultiHot: Boolean = false): (Double, (Array[Float], Array[Float])) = {

    val targets = batchIdx.flatMap(_._2)

    var startTime = System.currentTimeMillis()
    val params = makeParams(batchIdx.map(_._1), userNumSample, itemNumSample, userFeatureDim, itemFeatureDim, model,
      graphType, true, userFieldNum, itemFieldNum, fieldMultiHot)
    var weights = model.readWeights()
    params.put("weights", weights)
    params.put("targets", targets)
    val makeParamTime = System.currentTimeMillis() - startTime

    startTime = System.currentTimeMillis()
    val loss = torch.gcnBackward(params, userFieldNum > 0)
    val gcnBackwardTime = System.currentTimeMillis() - startTime

    startTime = System.currentTimeMillis()
    model.step(weights, optim)
    if (userFieldNum > 0) {
      //check if the grad really replaced the pulledUEmbedding
      val pulledUEmbedding = params.get("pulledUEmbedding").asInstanceOf[Array[Vector]]
      val uFeats = params.get("uFeats").asInstanceOf[Array[Int]]
      val uEmbeddingInput = params.get("u").asInstanceOf[Array[Float]]
      val embeddingDim = pulledUEmbedding.length
      makeEmbeddingGrad(uEmbeddingInput, pulledUEmbedding, uFeats, embeddingDim)
      model.asInstanceOf[SparseBiSAGEPSModel].updateUserEmbedding(pulledUEmbedding, optim)
      if (itemFeatureDim > 0) {
        val pulledIEmbedding = params.get("pulledIEmbedding").asInstanceOf[Array[Vector]]
        val iFeats = params.get("iFeats").asInstanceOf[Array[Int]]
        val iEmbeddingInput = params.get("i").asInstanceOf[Array[Float]]
        val embeddingDim = pulledIEmbedding.length
        makeEmbeddingGrad(iEmbeddingInput, pulledIEmbedding, iFeats, embeddingDim)
        model.asInstanceOf[SparseBiSAGEPSModel].updateItemEmbedding(pulledIEmbedding, optim)
      }
    }
    val updateTime = System.currentTimeMillis() - startTime
    println(s"partition $index, batchSize=${batchIdx.length}, makeParamTime=$makeParamTime, gcnBackWardTime=$gcnBackwardTime updateTime=$updateTime")

    val output = if (isSharedSamples) {
      weights = model.readWeights()
      params.remove("weights")
      params.put("weights", weights)
      predictBatch(params, targets, torch)
    } else {
      (new Array[Float](0), new Array[Float](0))
    }

    (loss, output)
  }

  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: BiSAGEPSModel,
                   userFeatureDim: Int,
                   itemFeatureDim: Int,
                   userNumSample: Int,
                   itemNumSample: Int,
                   isTest: Boolean,
                   userFieldNum: Int,
                   itemFieldNum: Int,
                   fieldMultiHot: Boolean): Iterator[(Array[Float], Array[Float])] = {

    val zips = if (isTest) {
      userTestIdx.zip(userTestLabels)
    } else {
      // randomly sample userTestIdx.length users
      val num = math.min(userTestIdx.length, userTrainIdx.length)
      val r = new Random()
      val start = r.nextInt(userTrainIdx.length - num + 1)
      userTrainIdx.zip(userTrainLabels).slice(start, start + num)
    }
    //    val zips = if (isTest) userTestIdx.zip(userTestLabels) else userTrainIdx.zip(userTrainLabels)
    val batchIterator = zips.sliding(batchSize, batchSize)
    val weights = model.readWeights()

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()

    new Iterator[(Array[Float], Array[Float])] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext)
          TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: (Array[Float], Array[Float]) = {
        val batch = batchIterator.next()
        predictBatch(batch, model, userFeatureDim, itemFeatureDim, userNumSample, itemNumSample, weights, torch,
          userFieldNum, itemFieldNum, fieldMultiHot)
      }
    }
  }

  def predictBatch(batchIdx: Array[(Int, Array[Float])],
                   model: BiSAGEPSModel,
                   userFeatureDim: Int,
                   itemFeatureDim: Int,
                   userNumSample: Int,
                   itemNumSample: Int,
                   weights: Array[Float],
                   torch: TorchModel,
                   userFieldNum: Int,
                   itemFieldNum: Int,
                   fieldMultiHot: Boolean): (Array[Float], Array[Float]) = {
    val targets = batchIdx.flatMap(f => f._2)
    val params = makeParams(batchIdx.map(f => f._1), userNumSample, itemNumSample, userFeatureDim, itemFeatureDim, model, 0, false, userFieldNum, itemFieldNum, fieldMultiHot)
    params.put("weights", weights)
    val outputs = torch.gcnPredict(params)
    outputs match {
      case f: Array[Float] => (targets, f)
      case l: Array[Long] => (targets, l.map(x => x.toFloat))
    }
  }

  def predictBatch(params: JMap[String, Object],
                   targets: Array[Float],
                   torch: TorchModel): (Array[Float], Array[Float]) = {
    val outputs = torch.gcnPredict(params)
    outputs match {
      case f: Array[Float] => (targets, f)
      case l: Array[Long] => (targets, l.map(x => x.toFloat))
    }
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

  def genLabelsEmbedding(batchSize: Int,
                         model: BiSAGEPSModel,
                         userFeatureDim: Int,
                         itemFeatureDim: Int,
                         userNumSample: Int,
                         itemNumSample: Int,
                         numPartitions: Int,
                         graphType: Int,
                         multiLabelsNum: Int,
                         userFieldNum: Int,
                         itemFieldNum: Int,
                         fieldMultiHot: Boolean): Iterator[(Long, String, String, String)] = {
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    val keysIterator = new Iterator[Array[(Long, String, String, String)]] with Serializable {
      override def hasNext: Boolean = {
        if (!batchIterator.hasNext) TorchModel.put(torch)
        batchIterator.hasNext
      }

      override def next: Array[(Long, String, String, String)] = {
        val batch = batchIterator.next().toArray
        val outputs = genLabelsEmbeddingBatch(batch, model, userFeatureDim, itemFeatureDim, userNumSample,
          itemNumSample, weights, torch, graphType, multiLabelsNum, userFieldNum, itemFieldNum, fieldMultiHot)
        outputs.toArray
      }
    }

    keysIterator.flatMap(f => f.iterator)
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

  def genLabelsEmbeddingBatch(batchIdx: Array[Int],
                              model: BiSAGEPSModel,
                              userFeatureDim: Int,
                              itemFeatureDim: Int,
                              userNumSample: Int,
                              itemNumSample: Int,
                              weights: Array[Float],
                              torch: TorchModel,
                              graphType: Int,
                              multiLabelsNum: Int,
                              userFieldNum: Int,
                              itemFieldNum: Int,
                              fieldMultiHot: Boolean): Iterator[(Long, String, String, String)] = {
    val batchIds = batchIdx.map(idx => keys(idx))
    var params = makeParams(batchIdx, userNumSample, itemNumSample, userFeatureDim, itemFeatureDim, model, graphType, false, userFieldNum, itemFieldNum, fieldMultiHot)
    params.put("weights", weights)
    val embedding = torch.gcnEmbedding(params)
    assert(embedding.length % batchIds.length == 0)
    params = makeParams(embedding, batchIds.length, weights)
    val outputs = torch.gcnPred(params)
    assert(outputs.length % batchIds.length == 0)
    val numLabels = outputs.length / batchIds.length
    val outputDim = embedding.length / batchIdx.length
    outputs.sliding(numLabels, numLabels).zip(embedding.sliding(outputDim, outputDim))
      .zip(batchIds.iterator).map {
      case ((p, h), key) =>
        val outLabels = if (multiLabelsNum == 1) {
          val temp = if (p.length == 1)
            if (p(0) >= 0.5) 1L
            else 0L
          else p.zipWithIndex.maxBy(_._1)._2
          temp.toString
        } else {
          p.map(x => if (x >= 0.5) 1 else 0).mkString(",")
        }

        (key, outLabels, h.mkString(","), p.mkString(","))
    }
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
    val edgeTypesList = new IntArrayList()
    val itemTypesList = new IntArrayList()

    for (idx <- batchIdx)
      srcIndex.put(keys(idx), srcIndex.size())

    val (first, second, second_, eTypes, iTypes, eSecondTypes, eSecondTypes_, iSecondTypes) =
      MakeBiEdgeIndex.makeEdgeIndex(batchIdx, keys, indptr, neighbors, edgeTypes, dstTypes, srcs,
        dsts, edgeTypesList, itemTypesList, srcIndex, dstIndex, userNumSample, itemNumSample, model, useSecondOrder,
        graphType, hasEdgeType, hasNodeType)

    val params = new JHashMap[String, Object]()

    val (pos_u, uBatchIds, uFieldIds) = MakeSparseBiFeature.makeFeatures(srcIndex, userFeatureDim, model, 0, params, userFieldNum, fieldMultiHot)
    val (pos_i, iBatchIds, iFieldIds) = if (itemFeatureDim > 0)
      MakeSparseBiFeature.makeFeatures(dstIndex, itemFeatureDim, model, 1, params, itemFieldNum, fieldMultiHot) else (null, null, null)

    // sample u-i-u
    params.put("first_u_edge_index", first)
    if (hasEdgeType && eTypes != null && eTypes.length != 0) {
      params.put("first_u_edge_type", eTypes)
    }
    if (hasNodeType && iTypes != null && iTypes.length != 0) {
      params.put("first_u_edge_i_type", iTypes)
    }
    if (useSecondOrder) {
      params.put("first_i_edge_index", second)
      params.put("second_u_edge_index", second_)
      if (hasEdgeType) {
        params.put("first_i_edge_type", eSecondTypes)
        params.put("second_u_edge_type", eSecondTypes_)
      }
      if (hasNodeType) params.put("second_u_edge_i_type", iSecondTypes)
    }

    params.put("u", pos_u)
    if (userFieldNum > 0) {
      params.put("u_batch_ids", uBatchIds)
      params.put("u_field_ids", uFieldIds)
    }
    if (itemFeatureDim > 0) {
      params.put("i", pos_i)
      if (itemFieldNum > 0) {
        params.put("i_batch_ids", iBatchIds)
        params.put("i_field_ids", iFieldIds)
      }
    }

    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(userFeatureDim))
    params.put("user_feature_dim", new Integer(userFeatureDim))
    if (itemFeatureDim > 0)
      params.put("item_feature_dim", new Integer(itemFeatureDim))
    params
  }

  def makeParams(embedding: Array[Float],
                 batchSize: Int,
                 weights: Array[Float]): JMap[String, Object] = {
    val hiddenDim = embedding.length / batchSize
    val params = new JHashMap[String, Object]()
    params.put("embedding", embedding)
    params.put("weights", weights)
    params.put("hidden_dim", new Integer(hiddenDim))
    params.put("feature_dim", new Integer(0))
    params
  }
}