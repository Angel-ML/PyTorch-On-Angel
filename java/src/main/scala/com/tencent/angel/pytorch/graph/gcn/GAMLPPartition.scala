package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.ml.math2.vector.IntFloatVector
import com.tencent.angel.pytorch.graph.gcn.MakeSparseBiFeature.makeFeatures

import java.util.{HashMap => JHashMap, Map => JMap}

class GAMLPPartition(index: Int,
                     keys: Array[Long],
                     features: Array[IntFloatVector],
                     trainIdx: Array[Int],
                     trainLabels: Array[Array[Float]],
                     testIdx: Array[Int],
                     testLabels: Array[Array[Float]],
                     torchModelPath: String) extends
  GCNPartition(index, keys, null, null,
    trainIdx, trainLabels, testIdx, testLabels, torchModelPath, false) {

  override
  def makeParams(batchIdx: Array[Int],
                 numSample: Int,
                 featureDim: Int,
                 model: GNNPSModel,
                 isTraining: Boolean,
                 fieldNum: Int,
                 fieldMultiHot: Boolean): JMap[String, Object] = {
    
    val x = makeFeatures(featureDim, batchIdx.map(i => features(i)), true)

    val params = new JHashMap[String, Object]()
    params.put("x", x)
    params.put("batch_size", new Integer(batchIdx.length))
    params.put("feature_dim", new Integer(featureDim))
    params
  }

}
