package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.ml.math2.storage.{IntFloatDenseVectorStorage, IntFloatSortedVectorStorage}
import com.tencent.angel.ml.math2.vector.IntFloatVector
import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap

object MakeFeature {

  def makeFeatures(index: Long2IntOpenHashMap, featureDim: Int, model: GNNPSModel): Array[Float] = {
    val size = index.size()
    val x = new Array[Float](size * featureDim)
    val keys = index.keySet().toLongArray
    val features = model.getFeatures(keys)
    //    assert(features.size() == keys.length)
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

  def sampleFeatures(size: Int, featureDim: Int, model: GNNPSModel): Array[Float] = {
    val x = new Array[Float](size * featureDim)
    val features = model.sampleFeatures(size)
    for (idx <- 0 until size)
      makeFeature(idx * featureDim, features(idx), x)
    x
  }


}
