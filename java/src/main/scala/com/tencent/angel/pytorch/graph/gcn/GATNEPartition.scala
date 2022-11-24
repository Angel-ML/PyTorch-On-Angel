package com.tencent.angel.pytorch.graph.gcn
import com.tencent.angel.graph.client.psf.sample.sampleneighbor.SampleMethod
import com.tencent.angel.graph.data.{EmbeddingOrGrad, FeatureFormat}
import com.tencent.angel.ml.math2.vector.IntIntVector
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, Long2ObjectOpenHashMap}
import com.tencent.angel.ps.storage.vector.element.IElement
import com.tencent.angel.spark.models.PSVector
import it.unimi.dsi.fastutil.ints.{Int2ObjectOpenHashMap, IntArrayList}
import it.unimi.dsi.fastutil.longs.{LongArrayList}

import java.util.{HashMap => JHashMap, Map => JMap}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import Array._
import scala.util.Random


class GATNEPartition(index: Int,
                     pairs: Map[Int, Array[(Long, Long, Int)]],   // src, dst, edge_Type
                     torchModelPath: String,
                     dataFormat: FeatureFormat) extends Serializable {
  var totalCallNum: Long = 0

  def trainEpoch(model: EmbeddingGNNPSModel,
                 batchSize: Int,
                 optim: AsyncOptim,
                 all_nodeTypes: Array[Int],
                 all_edgeTypes: Array[Int],
                 schema: Map[(Int, Int), Int],
                 featureDimsMap: Map[Int, Int],
                 embedDimsMap: Map[Int, Int],
                 fieldNumsMap: Map[Int, Int],
                 featureSplitIdxsMap: Map[Int, Int],
                 fieldMultiHot: Boolean,
                 contextDim: Int,
                 numNegative: Int,
                 numSampleMap: Map[Int, Int],
                 sampleMethod: SampleMethod,
                 logStep: Int,
                 localSample: Boolean,
                 negSampleByNodeType: Boolean,
                 maxIndex: Int,
                 maxIndexMap: mutable.Map[Int, Int],
                 minIndexMap: mutable.Map[Int, Int]): (Double, Int, Int) = {
    var lossSum = 0.0
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    var numStep = 0
    all_nodeTypes.foreach( type_id => {
      val batchIterator = pairs.getOrElse(type_id, Array[(Long, Long, Int)]()).sliding(batchSize, batchSize)
      while (batchIterator.hasNext) {
        val batch = batchIterator.next()
        val data = GATNEPartition.parseBatchData(model, batch, numNegative, localSample, negSampleByNodeType, maxIndex, maxIndexMap, minIndexMap)
        val loss = trainBatch(data._1, data._2, data._3, data._4, model, type_id, optim, torch, all_nodeTypes, all_edgeTypes, schema, featureDimsMap,
          embedDimsMap, fieldNumsMap, featureSplitIdxsMap, fieldMultiHot, contextDim, numNegative, numSampleMap, sampleMethod, logStep)
        lossSum += loss // * batch.length
        numStep += 1
      }
    })
    TorchModel.put(torch)
    (lossSum, numStep, numStep)
  }

  def incCallNum(): Unit = {
    totalCallNum = totalCallNum + 1
  }

  def logFlag(step: Int): Boolean = {
    totalCallNum % step == 0
  }

  def makeEmbeddingGrads(grad: Array[Float], featIds: Array[Long], embeddingDim: Int): Long2ObjectOpenHashMap[Array[Float]] = {
    val counts = featIds.map(f => (f, 1)).groupBy(f => f._1).map(f => (f._1, f._2.length))
    val grads = new Long2ObjectOpenHashMap[Array[Float]](counts.size)

    for (i <- featIds.indices) {
      val idx = featIds(i)
      if (grads.get(idx) == null) grads.put(idx, new Array[Float](embeddingDim))
      grads.put(idx, grads.get(idx).zipWithIndex.map(p => p._1 + grad(i * embeddingDim + p._2)))
    }

    for (pair <- counts) {
      grads.put(pair._1, grads.get(pair._1).map(f => f / pair._2.toFloat))
    }

    grads
  }

  def trainBatch(srcNodes: Array[Long],
                 dstNodes: Array[Long],
                 edgeTypes: Array[Int],
                 negs: Array[Array[Long]],
                 model: EmbeddingGNNPSModel,
                 src_type: Int,
                 optim: AsyncOptim,
                 torch: TorchModel,
                 all_nodeTypes: Array[Int],
                 all_edgeTypes: Array[Int],
                 schema: Map[(Int, Int), Int],
                 featureDimsMap: Map[Int, Int],
                 embedDimsMap: Map[Int, Int],
                 fieldNumsMap: Map[Int, Int],
                 featureSplitIdxsMap: Map[Int, Int],
                 fieldMultiHot: Boolean,
                 contextDim: Int,
                 numNegative: Int,
                 numSampleMap: Map[Int, Int],
                 sampleMethod: SampleMethod,
                 logStep: Int): Double = {
    incCallNum()
    val (params, context_x, dstIndex) = makeParams(srcNodes, dstNodes, src_type, edgeTypes, negs, featureDimsMap,
      embedDimsMap, fieldNumsMap, featureSplitIdxsMap, fieldMultiHot, model, all_nodeTypes, all_edgeTypes, schema,
      contextDim, numNegative, numSampleMap, sampleMethod)
    val weights = model.readWeights()
    params.put("weights", weights)
    params.put("src_type", new Integer(src_type))

    val loss = torch.gatneBackward(params, fieldNumsMap.nonEmpty)

    val context_x_grad = context_x.sliding(contextDim, contextDim).toArray
    val nodeId = new Array[Long](context_x_grad.length)
    val grads = new Array[EmbeddingOrGrad](context_x_grad.length)

    val it = dstIndex.long2IntEntrySet().fastIterator()
    while (it.hasNext) {
      val entry = it.next()
      val (dst, idx) = (entry.getLongKey, entry.getIntValue)
      nodeId(idx) = dst
      grads(idx) = new EmbeddingOrGrad(context_x_grad(idx))
    }
    model.step(weights, optim)
    model.embeddingStep(nodeId, grads.map(_.asInstanceOf[IElement]), optim)

    //check if the grad really replaced the pulledUEmbedding
    if (fieldNumsMap.nonEmpty) {
      all_nodeTypes.foreach{ type_id =>
        val feats = params.get(type_id + "Feats").asInstanceOf[Array[Long]]
        val FeatGrads = params.get("feats").asInstanceOf[Array[Array[Float]]](type_id)
        val embeddingDim = embedDimsMap.getOrElse(type_id, -1)
        if (feats != null) {
          val grads = makeEmbeddingGrads(FeatGrads, feats, embeddingDim)
          model.asInstanceOf[EmbeddingGNNPSModel].updateEmbedding(grads, optim, type_id)
        }
      }
    }

    if (logFlag(logStep)) {
      println(s"srcNodesLen=${srcNodes.length} loss=$loss")
    }
    loss
  }

  def genEmbeddingEpoch(model: EmbeddingGNNPSModel,
                        batchSize: Int,
                        all_nodeTypes: Array[Int],
                        all_edgeTypes: Array[Int],
                        schema: Map[(Int, Int), Int],
                        featureDimsMap: Map[Int, Int],
                        embedDimsMap: Map[Int, Int],
                        fieldNumsMap: Map[Int, Int],
                        featureSplitIdxsMap: Map[Int, Int],
                        fieldMultiHot: Boolean,
                        contextDim: Int,
                        numNegative: Int,
                        numSampleMap: Map[Int, Int],
                        sampleMethod: SampleMethod): Array[(Long, Int, Array[Float])] = {
    val weights = model.readWeights()
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()

    all_nodeTypes.flatMap( type_id => {
      val batchIterator = pairs.getOrElse(type_id, Array[(Long, Long, Int)]()).sliding(batchSize, batchSize)
      val keyIterator = new Iterator[Array[(Long, Int, Array[Float])]] with Serializable {
        override def hasNext: Boolean = {
          if (!batchIterator.hasNext) TorchModel.put(torch)
          batchIterator.hasNext
        }

        override def next: Array[(Long, Int, Array[Float])] = {
          val batch = batchIterator.next().toArray
          val output = genEmbeddingBatch(batch, model, type_id, weights, torch, all_nodeTypes, all_edgeTypes, schema,
            featureDimsMap, embedDimsMap, fieldNumsMap, featureSplitIdxsMap, fieldMultiHot, contextDim, numNegative, numSampleMap, sampleMethod)
          output.toArray
        }
      }
      keyIterator.flatMap(f => f.iterator)
    })
  }

  def genEmbeddingBatch(batch: Array[(Long, Long, Int)],
                        model: EmbeddingGNNPSModel,
                        src_type: Int,
                        weights: Array[Float],
                        torch: TorchModel,
                        all_nodeTypes: Array[Int],
                        all_edgeTypes: Array[Int],
                        schema: Map[(Int, Int), Int],
                        featureDimsMap: Map[Int, Int],
                        embedDimsMap: Map[Int, Int],
                        fieldNumsMap: Map[Int, Int],
                        featureSplitIdxsMap: Map[Int, Int],
                        fieldMultiHot: Boolean,
                        contextDim: Int,
                        numNegative: Int,
                        numSampleMap: Map[Int, Int],
                        sampleMethod: SampleMethod): Iterator[(Long, Int, Array[Float])] = {
    val srcNodes = batch.map(f => f._1)
    val dstNodes = batch.map(f => f._2)
    val edgeTypes = batch.map(f => f._3)
    val (params, _, _) = makeParams(srcNodes, dstNodes, src_type, edgeTypes, null, featureDimsMap, embedDimsMap,
      fieldNumsMap, featureSplitIdxsMap, fieldMultiHot, model, all_nodeTypes, all_edgeTypes, schema, contextDim,
      numNegative, numSampleMap, sampleMethod, false)

    params.put("weights", weights)
    params.put("src_type", new Integer(src_type))

    val output = torch.gcnEmbedding(params, fieldNumsMap.nonEmpty)
    assert(output.length % srcNodes.length == 0)
    val outputDim = output.length / srcNodes.length
    output.sliding(outputDim, outputDim)
      .zip(srcNodes.zip(edgeTypes).iterator).map {
      case (h, key) => // h is representation for node ``key``
        (key._1, key._2, h)
    }
  }

  def makeParams(srcNodes: Array[Long],
                 dstNodes: Array[Long],
                 src_type: Int,
                 edgeTypes: Array[Int],
                 negs: Array[Array[Long]],
                 featureDimsMap: Map[Int, Int],
                 embedDimsMap: Map[Int, Int],
                 fieldNumsMap: Map[Int, Int],
                 featureSplitIdxsMap: Map[Int, Int],
                 fieldMultiHot: Boolean,
                 model: EmbeddingGNNPSModel,
                 all_nodeTypes: Array[Int],
                 all_edgeTypes: Array[Int],
                 schema: Map[(Int, Int), Int],
                 contextDim: Int,
                 numNegative: Int,
                 numSampleMap: Map[Int, Int],
                 sampleMethod: SampleMethod,
                 training: Boolean = true): (JMap[String, Object], Array[Float], Long2IntOpenHashMap) = {
    val nodeIndexsMap = new mutable.HashMap[Int, Long2IntOpenHashMap]()
    all_nodeTypes.foreach(name => nodeIndexsMap.put(name, new Long2IntOpenHashMap()))
    val params = new JHashMap[String, Object]()

    val srcIndex = nodeIndexsMap.getOrElse(src_type, new Long2IntOpenHashMap())
    for (src <- srcNodes.distinct) {
      srcIndex.put(src, srcIndex.size())
    }
    nodeIndexsMap.update(src_type, srcIndex)

    val (srcNodesIndex, neighborIndex, neighborNum, neighborType, neighborFlag) = MakeEdgeIndex.makeNeighborIndex(srcNodes, src_type, nodeIndexsMap,
      numSampleMap, model, sampleMethod, all_edgeTypes, schema)
    // pull features for each type neighbor
    val featsMap = new ArrayBuffer[Array[Float]]()
    val featsDenseMap = new ArrayBuffer[Array[Float]]()
    val batchIdsMap = new ArrayBuffer[Array[Int]]()
    val fieldIdsMap = new ArrayBuffer[Array[Int]]()
    val featureSplitIdxs = new ArrayBuffer[Int]()
    val featDims = new ArrayBuffer[Int]()
    val embedDims = new ArrayBuffer[Int]()

    all_nodeTypes.foreach( type_id => {
      val index = nodeIndexsMap.getOrElse(type_id,  null)
      val featureDim = featureDimsMap.getOrElse(type_id, -1)
      val fieldDim = fieldNumsMap.getOrElse(type_id, -1)
      val embedDim = embedDimsMap.getOrElse(type_id, -1)
      val splitIdx = featureSplitIdxsMap.getOrElse(type_id, -1)

      val (feat, feat_dense, b, f) = if (index.size() != 0)
        MakeSparseBiFeature.makeFeatures(index, featureDim, model, type_id, params, fieldDim, fieldMultiHot, embedDim, dataFormat, splitIdx)
      else (new Array[Float](0), new Array[Float](0), new Array[Int](0), new Array[Int](0))

      featsMap.append(feat)
      if (fieldDim > 0) {
        batchIdsMap.append(b)
        fieldIdsMap.append(f)
      }
      if (dataFormat == FeatureFormat.DENSE_HIGH_SPARSE) {
        featsDenseMap.append(feat_dense)
        featureSplitIdxs.append(splitIdx)
      }
      featDims.append(featureDim)
      embedDims.append(embedDim)
    })

    val dstIndex = new Long2IntOpenHashMap()
    var context_x: Array[Float] = new Array[Float](0)
    if (training) {
      for(dst <- dstNodes.distinct) {
        dstIndex.put(dst, dstIndex.size())
      }

      val (dstNodesIndex, negsIndex) = MakeEdgeIndex.makeNegativeIndex(dstNodes, negs, dstIndex)
      val dstFeats = model.getContext(dstNodes, negs)
      context_x = MakeSparseBiFeature.makeFeatures_(dstIndex, contextDim, dstFeats)
      params.put("dsts", dstNodesIndex)
      params.put("context_x", context_x)
      params.put("negatives", negsIndex)
      params.put("context_dim", new Integer(contextDim))
      params.put("negative_num", new Integer(numNegative))
    }

    params.put("srcs", srcNodesIndex)
    params.put("first_edge_type", edgeTypes.map(_.toLong))
    params.put("feats", featsMap.toArray)
    params.put("feature_dims", featDims.toArray)
    params.put("neighbors", neighborIndex)
    params.put("neighbors_type", neighborType)
    params.put("neighbors_flag", neighborFlag)
    params.put("embedding_dims", embedDims.toArray)
    params.put("neighbor_num", neighborNum)

    params.put("edge_type_num", new Integer(all_edgeTypes.length))
    params.put("batch_size", new Integer(srcNodes.length))

    if (fieldNumsMap.nonEmpty) {
      params.put("batchIds", batchIdsMap.toArray)
      params.put("fieldIds", fieldIdsMap.toArray)
      params.put("batchIds_size", new Integer(batchIdsMap.length))
    }
    if (dataFormat == FeatureFormat.DENSE_HIGH_SPARSE) {
      params.put("feats_dense", featsDenseMap.toArray)
      params.put("feature_dense_dims", featureSplitIdxs.toArray)
    }

    (params, context_x, dstIndex)
  }

}

private[gcn]
object GATNEPartition {
  def apply(index: Int,
            iterator: Iterator[(Long, Long, Int, Int)],
            torchModelPath: String,
            dataFormat: FeatureFormat): GATNEPartition = {
    val pairs = new mutable.HashMap[Int, ArrayBuffer[(Long, Long, Int)]]()
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (src, dst, src_type, edge_type) = (entry._1, entry._2, entry._3, entry._4)
      val pair = pairs.getOrElse(src_type, new ArrayBuffer[(Long, Long, Int)]())
      pair.append((src, dst, edge_type))
      pairs.update(src_type, pair)
    }
    new GATNEPartition(index, pairs.map(p => (p._1, p._2.toArray)).toMap, torchModelPath, dataFormat)
  }

  def parseBatchData(model: EmbeddingGNNPSModel,
                     sentences: Array[(Long, Long, Int)],
                     negative: Int,
                     localSample: Boolean,
                     negSampleByNodeType: Boolean,
                     maxIndex: Int,
                     maxIndexMap: mutable.Map[Int, Int],
                     minIndexMap: mutable.Map[Int, Int],
                     seed: Int = Random.nextInt): (Array[Long], Array[Long], Array[Int], Array[Array[Long]]) = {
    val rand = new Random(seed)
    val (srcNodes, dstNodes, edgeTypes) = sentences.unzip3
    val negativeSamples = if (negSampleByNodeType) {
      if (localSample) {
        localNegativeSampleWithType(model, srcNodes, dstNodes, negative, maxIndex, maxIndexMap, minIndexMap, rand.nextInt())
      } else {
        val index = negativeSampleWithType(model, dstNodes, negative, maxIndex, maxIndexMap, minIndexMap, rand.nextInt())
        val negs_nodeIndex = index.flatten.distinct
        val index2node = model.readIndex2Node(negs_nodeIndex)
        index.map(i => index2node.get(i))
      }
    } else {
      if (localSample) {
        localNegativeSample(model, srcNodes, dstNodes, negative, maxIndex, 0, rand.nextInt())
      } else {
        val index = negativeSample(dstNodes, negative, maxIndex, 0, rand.nextInt())
        val negs_nodeIndex = index.flatten.distinct
        val index2node = model.readIndex2Node(negs_nodeIndex)
        index.map(i => index2node.get(i))
      }
    }
    (srcNodes, dstNodes, edgeTypes, negativeSamples)
  }

  def negativeSample(dstNodes: Array[Long],
                     sampleNum: Int,
                     maxIndex: Int,
                     minIndex: Int,
                     seed: Int): Array[Array[Int]] = {
    val rand = new Random(seed)
    val sampleWords = new Array[Array[Int]](dstNodes.length)
    var wordIndex: Int = 0

    for (i <- dstNodes.indices) {
      var sampleIndex: Int = 0
      sampleWords(wordIndex) = new Array[Int](sampleNum)
      while (sampleIndex < sampleNum) {
        val target = rand.nextInt(maxIndex - minIndex) + minIndex
        if (target != dstNodes(i)) {
          sampleWords(wordIndex)(sampleIndex) = target
          sampleIndex += 1
        }
      }
      wordIndex += 1
    }
    sampleWords
  }

  def localNegativeSample(model: EmbeddingGNNPSModel,
                          srcNodes: Array[Long],
                          dstNodes: Array[Long],
                          sampleNum: Int,
                          maxIndex: Int,
                          minIndex: Int,
                          seed: Int): Array[Array[Long]] = {
    val rand = new Random(seed)
    val sampleWords = new Array[Array[Long]](srcNodes.length)

    var nodes = new mutable.HashSet[Long]()
    nodes ++= srcNodes
    nodes ++= dstNodes
    val sampleSet = nodes.toArray

    for (i <- srcNodes.indices) {
      var sampleIndex: Int = 0
      val len = sampleSet.length
      sampleWords(i) = new Array[Long](sampleNum)
      if (len < 3) { // in case local sampling is invalid
        val sampleWords_index = new Array[Int](sampleNum)
        while (sampleIndex < sampleNum) {
          val target = rand.nextInt(maxIndex - minIndex) + minIndex
          if (target != srcNodes(i) && target != dstNodes(i)) {
            sampleWords_index(sampleIndex) = target
            sampleIndex += 1
          }
        }
        val index2node = model.readIndex2Node(sampleWords_index)
        sampleWords(i) = sampleWords_index.map(i => index2node.get(i))
      } else {
        while (sampleIndex < sampleNum) {
          val target = sampleSet(rand.nextInt(len))
          if (target != srcNodes(i) && target != dstNodes(i)) {
            sampleWords(i)(sampleIndex) = target
            sampleIndex += 1
          }
        }
      }
    }
    sampleWords
  }

  def negativeSampleWithType(model: EmbeddingGNNPSModel,
                             dstNodes: Array[Long],
                             sampleNum: Int,
                             maxIndex: Int,
                             maxIndexMap: mutable.Map[Int, Int],
                             minIndexMap: mutable.Map[Int, Int],
                             seed: Int): Array[Array[Int]] = {
    val rand = new Random(seed)
    val nodes = dstNodes.distinct
    val types = model.readNodeTypes(nodes).get(nodes)
    val node2Type = nodes.zip(types).toMap
    val sampleWords = new Array[Array[Int]](dstNodes.length)

    for (i <- dstNodes.indices) {
      var sampleIndex: Int = 0
      sampleWords(i) = new Array[Int](sampleNum)
      val dstNodeType = node2Type(dstNodes(i))
      val max = maxIndexMap.getOrElse(dstNodeType, maxIndex)
      val min = minIndexMap.getOrElse(dstNodeType, 0)
      while (sampleIndex < sampleNum) {
        val target = rand.nextInt(max - min) + min
        if (target != dstNodes(i)) {
          sampleWords(i)(sampleIndex) = target
          sampleIndex += 1
        }
      }
    }
    sampleWords
  }

  def localNegativeSampleWithType(model: EmbeddingGNNPSModel,
                                  srcNodes: Array[Long],
                                  dstNodes: Array[Long],
                                  sampleNum: Int,
                                  maxIndex: Int,
                                  maxIndexMap: mutable.Map[Int, Int],
                                  minIndexMap: mutable.Map[Int, Int],
                                  seed: Int): Array[Array[Long]] = {
    val rand = new Random(seed)
    var nodes = new mutable.HashSet[Long]()
    nodes ++= srcNodes
    nodes ++= dstNodes
    val sampleSet = nodes.toArray

    val types = model.readNodeTypes(sampleSet).get(sampleSet)
    val node2Type = sampleSet.zip(types).toMap
    val type2NodeList = new Int2ObjectOpenHashMap[LongArrayList]()
    types.zip(sampleSet).map{ case (nodeType, nodeId) =>
      if(!type2NodeList.containsKey(nodeType)) {
        val nodeSet = new LongArrayList()
        nodeSet.add(nodeId)
        type2NodeList.put(nodeType, nodeSet)
      } else {
        type2NodeList.get(nodeType).add(nodeId)
      }
    }
    val sampleWords = new Array[Array[Long]](dstNodes.length)

    for (i <- dstNodes.indices) {
      var sampleIndex: Int = 0
      sampleWords(i) = new Array[Long](sampleNum)
      val dstNodeType = node2Type(dstNodes(i))
      val dstNodeSet = type2NodeList.get(dstNodeType)
      val max = maxIndexMap.getOrElse(dstNodeType, maxIndex)
      val min = minIndexMap.getOrElse(dstNodeType, 0)

      val len = dstNodeSet.size
      if (len < 3) { // in case local sampling is invalid
        val sampleWords_index = new Array[Int](sampleNum)
        while (sampleIndex < sampleNum) {
          val target = rand.nextInt(max - min) + min
          if (target != srcNodes(i) && target != dstNodes(i)) {
            sampleWords_index(sampleIndex) = target
            sampleIndex += 1
          }
        }
        val index2node = model.readIndex2Node(sampleWords_index)
        sampleWords(i) = sampleWords_index.map(i => index2node.get(i))
      } else {
        while (sampleIndex < sampleNum) {
          val target = dstNodeSet.getLong(rand.nextInt(dstNodeSet.size()))
          if (target != dstNodes(i)) {
            sampleWords(i)(sampleIndex) = target
            sampleIndex += 1
          }
        }
      }
    }
    sampleWords
  }
}