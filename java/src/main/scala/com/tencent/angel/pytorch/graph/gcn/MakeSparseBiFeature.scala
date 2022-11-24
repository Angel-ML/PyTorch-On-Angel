/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
package com.tencent.angel.pytorch.graph.gcn

import java.util

import com.tencent.angel.ml.math2.storage.{IntFloatDenseVectorStorage, IntFloatSortedVectorStorage}
import com.tencent.angel.ml.math2.vector.{IntFloatVector, Vector}
import it.unimi.dsi.fastutil.floats.FloatArrayList
import it.unimi.dsi.fastutil.ints.{IntArrayList, IntOpenHashSet}
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, Long2ObjectOpenHashMap}

object MakeSparseBiFeature {

  def makeFeatures(index: Long2IntOpenHashMap, featureDim: Int, model: GNNPSModel,
                   graphType: Int, params: util.HashMap[String, Object],
                   fieldNum: Int, fieldMultiHot: Boolean = false): (Array[Float], Array[Int], Array[Int]) = {
    val keys = index.keySet().toLongArray
    val features = model match {
      case biModel: BiSAGEPSModel => biModel.getFeatures(keys, graphType)
      case gnnModel : GNNPSModel => gnnModel.getFeatures(keys)
    }
    //    println(s"numKeys vs numHasFeatures: ${keys.length} vs ${features.size()}")
    val temp = features.keySet().toLongArray
    //    assert(features.size() == keys.length)
    features.get(temp(0)).getStorage match {
      case _: IntFloatDenseVectorStorage =>
        (makeFeatures(index, featureDim, features), null, null)
      case _: IntFloatSortedVectorStorage =>
        if (fieldNum > 0) {
          makeSparseFeatures(index, featureDim, model, graphType, features, params, fieldNum, fieldMultiHot)
        } else {
          (makeFeatures(index, featureDim, features), null, null)
        }
    }
  }
  
  def makeFeatures_(index: Long2IntOpenHashMap, featureDim: Int,
                    features: Long2ObjectOpenHashMap[Array[Float]]): Array[Float] = {
    val size = index.size()
    val x = new Array[Float](size * featureDim)
    val temp = features.keySet().toLongArray
    if (features.size() != size) {
      println(s"context numKeys vs numHasFeatures: ${size} vs ${features.size()}")
      index.keySet().toLongArray.diff(temp).foreach(println)
    }
    assert(features.size() == size)
    val it = features.long2ObjectEntrySet().fastIterator()
    while (it.hasNext) {
      val entry = it.next()
      val (node, f) = (entry.getLongKey, entry.getValue)
      val start = index.get(node) * featureDim
      var j = 0
      while (j < featureDim) {
        x(start + j) = f(j)
        j += 1
      }
    }
    x
  }

  def makeFeatures(index: Long2IntOpenHashMap, featureDim: Int,
                   features: Long2ObjectOpenHashMap[IntFloatVector]): Array[Float] = {
    val size = index.size()
    val x = new Array[Float](size * featureDim)
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
    val values = f.getStorage.getValues
    var j = 0
    while (j < values.length) {
      x(start + j) = values(j)
      j += 1
    }
  }

  def sampleFeatures(size: Int, featureDim: Int, model: GNNPSModel, graphType: Int,
                     dataFormat: String, params: util.HashMap[String, Object], fieldNum: Int,
                     fieldMultiHot: Boolean = false): (Array[Float], Array[Int], Array[Int]) = {
    val x = new Array[Float](size * featureDim)
    val features = model match {
      case biModel: BiSAGEPSModel => biModel.sampleFeatures(size, graphType)
      case gnnModel : GNNPSModel => gnnModel.sampleFeatures(size)
    }
    if (fieldNum < 0) {
      for (idx <- 0 until size){
        val f = if (features(idx) != null) features(idx) else {
          if (dataFormat == "dense") {
            new IntFloatVector(featureDim, new IntFloatDenseVectorStorage(featureDim))
          } else {
            new IntFloatVector(featureDim, new IntFloatSortedVectorStorage(featureDim))
          }
        }
        makeFeature(idx * featureDim, f, x)
      }
      (x, null, null)
    } else {
      makeSparseFeatures(size, featureDim, model, graphType, features, params, fieldNum, fieldMultiHot)

    }
  }

  def makeSparseFeatures(index: Long2IntOpenHashMap, featureDim: Int, model: GNNPSModel,
                         graphType: Int, features: Long2ObjectOpenHashMap[IntFloatVector],
                         params: util.HashMap[String, Object], fieldNum: Int,
                         isMulti: Boolean): (Array[Float], Array[Int], Array[Int]) = {

    val batchIds = new IntArrayList()
    val fieldIds = new IntArrayList()
    val featIds = new IntArrayList()
    val values = new FloatArrayList()
    val feats2pull = new IntOpenHashSet()
    val keys = index.keySet().toLongArray.map(q => (q, index.get(q))).sortWith(_._2 < _._2).map(_._1)
    val oneHotFieldNum = if (isMulti) fieldNum - 1 else fieldNum
    var count = 0

    keys.foreach{ key =>
      val feat = features.get(key)
      if (feat != null) {
        val f = feat.getStorage
        val indices = f.getIndices
        val value = f.getValues
        var i = 0
        while (i < oneHotFieldNum) {
          batchIds.add(count)
          fieldIds.add(i)
          featIds.add(indices(i))
          values.add(value(i))
          feats2pull.add(indices(i))
          i += 1
        }
        if (i < f.size()) {
          for (j <- (i until f.size())) {
            batchIds.add(count)
            fieldIds.add(i)
            featIds.add(indices(j))
            values.add(value(j))
            feats2pull.add(indices(j))
          }
          i += 1
        }
        count += 1
      } else {
        (0 until fieldNum).foreach{ i =>
          batchIds.add(count)
          fieldIds.add(i)
          featIds.add(0)
          values.add(0)
          feats2pull.add(0)
        }
        count += 1
      }
    }

    val pulledEmbeddings = model match {
      case biModel: SparseBiSAGEPSModel => {
        val embedding = biModel.getEmbedding(feats2pull.toIntArray, graphType)
        if (params != null) {
          val (embedName, featsName, embedDim) =
            if (graphType == 0) ("pulledUEmbedding", "uFeats", "u_embedding_dim")
            else ("pulledIEmbedding", "iFeats", "i_embedding_dim")
          params.put(embedName, embedding)
          params.put(featsName, featIds.toIntArray)
          params.put(embedDim, new Integer(embedding.length))
        }
        embedding
      }
      case gnnModel : SparseGNNPSModel => {
        val embedding = gnnModel.getEmbedding(feats2pull.toIntArray)
        if (params != null) {
          val (embedName, featsName, embedDim) = ("pulledEmbedding", "feats", "embedding_dim")
          params.put(embedName, embedding)
          params.put(featsName, featIds.toIntArray)
          params.put(embedDim, new Integer(embedding.length))
        }
        embedding
      }
    }

    val embeddingDim = pulledEmbeddings.length
    val embeddings = makeEmbedding(pulledEmbeddings, featIds, values, embeddingDim)

    (embeddings, batchIds.toIntArray, fieldIds.toIntArray)
  }

  def makeSparseFeatures(size: Int, featureDim: Int, model: GNNPSModel, graphType: Int,
                         features: Array[IntFloatVector],
                         params: util.HashMap[String, Object], fieldNum: Int,
                         isMulti: Boolean): (Array[Float], Array[Int], Array[Int])= {

    val batchIds = new IntArrayList()
    val fieldIds = new IntArrayList()
    val featIds = new IntArrayList()
    val values = new FloatArrayList()
    val oneHotFieldNum = if (isMulti) fieldNum - 1 else fieldNum
    val feats2pull = new IntOpenHashSet()
    var idx = 0
    for (feature <- features) {
      val f = if (feature == null) {
        new IntFloatVector(featureDim, new IntFloatSortedVectorStorage(featureDim))
      } else feature

      val indices = f.getStorage.getIndices
      val value = f.getStorage.getValues
      var i = 0

      while (i < oneHotFieldNum) {
        batchIds.add(idx)
        fieldIds.add(i)
        featIds.add(indices(i))
        values.add(value(i))
        feats2pull.add(indices(i))
        i += 1
      }
      if (i < f.size()) {
        for (j <- (i until f.size())) {
          batchIds.add(idx)
          fieldIds.add(i)
          featIds.add(indices(j))
          values.add(value(j))
          feats2pull.add(indices(j))
        }
        i += 1
      }
      idx += 1
    }

    val pulledEmbeddings = model match {
      case biModel: SparseBiSAGEPSModel => {
        biModel.getEmbedding(feats2pull.toIntArray, graphType)
      }
      case gnnModel : SparseGNNPSModel => {
        gnnModel.getEmbedding(feats2pull.toIntArray)
      }
    }
    val embeddingDim = pulledEmbeddings.length
    val embeddings = makeEmbedding(pulledEmbeddings, featIds, values, embeddingDim)

    (embeddings, batchIds.toIntArray, fieldIds.toIntArray)
  }

  def makeEmbedding(embedding: Array[Vector], featIds: IntArrayList, values: FloatArrayList,
                    embeddingDim: Int): Array[Float] = {
    val buf = new Array[Float](featIds.size() * embeddingDim)
    val ints = embedding.map(f => f.asInstanceOf[IntFloatVector])
    for (i <- 0 until featIds.size()) {
      for (j <- 0 until embeddingDim) {
        buf(i * embeddingDim + j) = ints(j).get(featIds.getInt(i)) * values.getFloat(i)
      }
    }

    buf
  }
}