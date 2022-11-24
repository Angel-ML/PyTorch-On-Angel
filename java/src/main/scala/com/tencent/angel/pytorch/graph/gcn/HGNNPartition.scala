package com.tencent.angel.pytorch.graph.gcn

import java.util.{Map => JMap}

import com.tencent.angel.graph.client.psf.sample.sampleneighbor.SampleMethod
import com.tencent.angel.graph.data.FeatureFormat
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap

class HGNNPartition(index: Int,
                    keys: Map[String, Array[Long]],
                    indptr: Map[String, Array[Int]],
                    neighbors: Map[String, Array[Long]],
                    weights: Map[String, Array[Float]],
                    aliasTables: Map[String, Map[Long, (Array[Float], Array[Int])]],
                    trainIdx: Map[String, Array[Int]],
                    torchModelPath: String,
                    useSecondOrder: Boolean,
                    dataFormat: FeatureFormat,
                    sampleMethod: SampleMethod) extends
  GNNPartition(index, null, null, null, torchModelPath, useSecondOrder, sampleMethod) {

  /**
    * make node id for edge
    * @param len: edge length
    * @param value1: src id
    * @param value2: dst id
    * @param value3: metaPath id
    * @return
    */
  def makeNodeTypeIds(len: Int, value1: Long, value2: Long, value3: Long): (Array[Long], Array[Long], Array[Long]) = {
    (Array.fill(len)(value1), Array.fill(len)(value2), Array.fill(len)(value3))
  }

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: HGNNPSModel,
                 featureDims: Map[String, Int],
                 optim: AsyncOptim,
                 numSamples:  Map[String, Int],
                 graphType: String,
                 fieldNums:  Map[String, Int],
                 fieldMultiHot: Boolean,
                 trainRatio: Float,
                 metaPaths: Array[String],
                 nodeNames: String,
                 filterSameNode: Boolean,
                 embedDims: Map[String, Int],
                 featureSplitIdxs: Map[String, Int]): (Double, Long, Int) = {
    val idxs = if (trainIdx != null && trainIdx.contains(graphType) && trainIdx.get(graphType).nonEmpty) {
      trainIdx.getOrElse(graphType, new Array[Int](0))
    } else keys.getOrElse(graphType, new Array[Long](0)).indices.toArray
    val batchIterator = sampleTrainData(idxs, trainRatio).sliding(batchSize, batchSize)

    var lossSum = 0.0
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0
    while (batchIterator.hasNext) {
      val batch = batchIterator.next().toArray
      val loss = trainBatch(batch, model, featureDims, optim, numSamples, torch, graphType,
        fieldNums, fieldMultiHot, metaPaths, nodeNames, filterSameNode, embedDims, featureSplitIdxs)
      lossSum += loss * batch.length
      numStep += 1
    }

    TorchModel.put(torch)
    (lossSum, idxs.length, numStep)
  }

  def trainBatch(batchIdx: Array[Int],
                 model: HGNNPSModel,
                 featureDimsMap: Map[String, Int],
                 optim: AsyncOptim,
                 numSamplesMap: Map[String, Int],
                 torch: TorchModel,
                 graphType: String,
                 fieldNumsMap: Map[String, Int],
                 fieldMultiHot: Boolean,
                 metaPaths: Array[String],
                 nodeNames: String,
                 filterSameNode: Boolean,
                 embedDims: Map[String, Int],
                 featureSplitIdxs: Map[String, Int]): Double = {

    val nNames = nodeNames.split(",")//total node type
    // recode nodes with different types
    val nodeNameIds = nNames.zipWithIndex.toMap

    val params = makeParams(batchIdx, numSamplesMap, featureDimsMap, model, graphType, true,
      fieldNumsMap, fieldMultiHot, metaPaths, nodeNameIds, filterSameNode, embedDims, featureSplitIdxs)

    val weights = model.readWeights()
    params.put("weights", weights)
    val loss = torch.gcnBackward(params, fieldNumsMap.size > 0)

    model.step(weights, optim)

    //check if the grad really replaced the pulledUEmbedding
    if (fieldNumsMap.size > 0) {
      nodeNameIds.foreach{ case (name, idx) =>
        val feats = params.get(name + "Feats").asInstanceOf[Array[Long]]
        val FeatGrads = params.get("feats").asInstanceOf[Array[Array[Float]]](idx)
        val embeddingDim = embedDims.getOrElse(name, -1)//pulledEmbedding.size()
        val grads = makeEmbeddingGrads(FeatGrads, feats, embeddingDim)
        model.asInstanceOf[HGNNPSModel].updateEmbedding(grads, optim, name)
      }
    }

    loss
  }

  def makeParams(batchIdx: Array[Int],
                 numSamplesMap: Map[String, Int],
                 featureDimsMap: Map[String, Int],
                 model: HGNNPSModel,
                 graphType: String,
                 isTraining: Boolean,
                 fieldNumsMap: Map[String, Int],
                 fieldMultiHot: Boolean,
                 metaPaths: Array[String],
                 nodeNameIds: Map[String, Int],
                 filterSameNode: Boolean,
                 embedDims: Map[String, Int],
                 featureSplitIdxs: Map[String, Int]): JMap[String, Object] = ???

  def makeParams(batchIdx: Array[Long],
                 numSamplesMap: Map[String, Int],
                 featureDimsMap: Map[String, Int],
                 model: HGNNPSModel,
                 graphType: String,
                 isTraining: Boolean,
                 fieldNumsMap: Map[String, Int],
                 fieldMultiHot: Boolean,
                 metaPaths: Array[String],
                 nodeNameIds: Map[String, Int],
                 filterSameNode: Boolean,
                 embedDims: Map[String, Int],
                 featureSplitIdxs: Map[String, Int]): JMap[String, Object] = ???


  def genEmbedding(batchSize: Int,
                   model: HGNNPSModel,
                   numSamplesMap: Map[String, Int],
                   featureDimsMap: Map[String, Int],
                   numPartitions: Int,
                   graphType: String,
                   fieldNums: Map[String, Int],
                   fieldMultiHot: Boolean,
                   metaPaths: Array[String],
                   nodeNames: String,
                   filterSameNode: Boolean,
                   embedDims: Map[String, Int],
                   featureSplitIdxs: Map[String, Int]): Iterator[(Long, String)] = {

    val nNames = nodeNames.split(",")//total node type
    // recode nodes with different types
    val nodeNameIds = nNames.zipWithIndex.toMap

    val batchIterator = keys.getOrElse(graphType, new Array[Long](0)).indices.sliding(batchSize, batchSize)
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
        val output = genEmbeddingBatch(batch, model, featureDimsMap, numSamplesMap, weights, torch, graphType, fieldNums, fieldMultiHot, metaPaths, nodeNameIds, filterSameNode, embedDims, featureSplitIdxs)
        output.toArray
      }
    }

    keyIterator.flatMap(f => f.iterator)
  }

  def genEmbeddingBatch(batchIdx: Array[Int],
                        model: HGNNPSModel,
                        numSamplesMap: Map[String, Int],
                        featureDimsMap: Map[String, Int],
                        weights: Array[Float],
                        torch: TorchModel,
                        graphType: String,
                        fieldNums: Map[String, Int],
                        fieldMultiHot: Boolean,
                        metaPaths: Array[String],
                        nodeNameIds: Map[String, Int],
                        filterSameNode: Boolean,
                        embedDims: Map[String, Int],
                        featureSplitIdxs: Map[String, Int]): Iterator[(Long, String)] = {

    val batchIds = batchIdx.map(idx => keys.getOrElse(graphType, null)(idx))
    val params = makeParams(batchIdx, featureDimsMap, numSamplesMap, model, graphType,
      false, fieldNums, fieldMultiHot, metaPaths, nodeNameIds, filterSameNode, embedDims, featureSplitIdxs)
    params.put("agg_node", new Integer(nodeNameIds.getOrElse(graphType, 0)))
    params.put("weights", weights)
    val output = torch.gcnEmbedding(params)
    assert(output.length % batchIdx.length == 0)
    val outputDim = output.length / batchIdx.length
    output.sliding(outputDim, outputDim)
      .zip(batchIds.iterator).map {
      case (h, key) => // h is representation for node ``key``
        (key, h.mkString(","))
    }
  }
}