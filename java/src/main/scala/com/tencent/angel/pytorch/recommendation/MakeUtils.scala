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
package com.tencent.angel.pytorch.recommendation

import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix
import com.tencent.angel.ml.math2.storage.{IntFloatDenseVectorStorage, IntFloatSparseVectorStorage, LongFloatSparseVectorStorage}
import com.tencent.angel.ml.math2.vector.{IntFloatVector, LongFloatVector, Vector}
import it.unimi.dsi.fastutil.ints.Int2FloatOpenHashMap
import it.unimi.dsi.fastutil.longs.Long2FloatOpenHashMap

object MakeUtils {

  /* make gradients and pytorch inputs */

  def makeBias(bias: Vector): Array[Float] =
    bias match {
      case int: IntFloatVector => makeBias(int)
      case long: LongFloatVector => makeBias(long)
    }

  def makeBias(bias: IntFloatVector): Array[Float] =
    bias.get(Array(0))

  def makeBias(bias: LongFloatVector): Array[Float] =
    null

  def makeWeight(weight: Vector, batch: CooLongFloatMatrix): Array[Float] = {
    weight match {
      case int: IntFloatVector =>
        val indices = batch.getColIndices
        val values = new Array[Float](indices.length)
        for (i <- indices.indices)
          values(i) = int.get(indices(i).toInt)
        values
      case long: LongFloatVector =>
        null
    }
  }

  def makeWeight(weight: Vector): Array[Float] = {
    weight match {
      case int: IntFloatVector =>
        val buf = new Array[Float](int.size())
        for (i <- 0 until int.size())
          buf(i) = int.get(i)
        buf
      case long: LongFloatVector =>
        null
    }
  }

  def makeEmbedding(embedding: Array[Vector], feats: Array[Long], embeddingDim: Int): Array[Float] = {
    val buf = new Array[Float](feats.length * embeddingDim)
    if (embedding(0).isInstanceOf[IntFloatVector]) {
      val ints = embedding.map(f => f.asInstanceOf[IntFloatVector])
      for (i <- feats.indices)
        for (j <- 0 until embeddingDim)
          buf(i * embeddingDim + j) = ints(j).get(feats(i).toInt)
    } else {
      val longs = embedding.map(f => f.asInstanceOf[LongFloatVector])
      for (i <- feats.indices)
        for (j <- 0 until embeddingDim)
          buf(i * embeddingDim + j) = longs(j).get(feats(i))
    }
    buf
  }

  def makeEmbedding(embedding: Array[Vector]): Array[Float] = {
    val buf = new Array[Float](embedding(0).getSize.toInt * embedding.length)
    val embeddingDim = embedding.length
    if (embedding(0).isInstanceOf[IntFloatVector]) {
      val ints = embedding.map(f => f.asInstanceOf[IntFloatVector])
      for (i <- 0 until ints(0).size())
        for (j <- 0 until embeddingDim)
          buf(i * embeddingDim + j) = ints(j).get(i)
    } else {
      val longs = embedding.map(f => f.asInstanceOf[LongFloatVector])
      for (i <- 0 until longs(0).size().toInt)
        for (j <- 0 until embeddingDim)
          buf(i * embeddingDim + j) = longs(j).get(i)
    }
    buf
  }

  def makeMats(mats: IntFloatVector): Array[Float] =
    mats.getStorage.asInstanceOf[IntFloatDenseVectorStorage].getValues

  def makeMats(mats: Vector): Array[Float] = makeMats(mats.asInstanceOf[IntFloatVector])

  def makeBiasGrad(grad: Array[Float], bias: IntFloatVector): Unit =
    bias.set(0, grad(0))

  def makeBiasGrad(grad: Array[Float], bias: Vector): Unit =
    makeBiasGrad(grad, bias.asInstanceOf[IntFloatVector])

  def makeWeightGrad(grad: Array[Float], weight: Vector, feats: Array[Long]): Unit = {
    weight match {
      case int: IntFloatVector =>
        val map = new Int2FloatOpenHashMap(int.size())
        for (i <- grad.indices)
          map.addTo(feats(i).toInt, grad(i))
        weight.setStorage(new IntFloatSparseVectorStorage(weight.dim().toInt, map))
      case long: LongFloatVector =>
        val map = new Long2FloatOpenHashMap(long.size().toInt)
        for (i <- grad.indices)
          map.addTo(feats(i), grad(i))
        weight.setStorage(new LongFloatSparseVectorStorage(weight.dim(), map))
    }
  }

  def makeEmbeddingGrad(grad: Array[Float], embedding: Array[Vector], feats: Array[Long], embeddingDim: Int): Unit = {
    if (embedding(0).isInstanceOf[IntFloatVector]) {
      val grads = embedding.map(f => new Int2FloatOpenHashMap(f.getSize.toInt))
      for (i <- feats.indices)
        for (j <- 0 until embeddingDim)
          grads(j).addTo(feats(i).toInt, grad(i * embeddingDim + j))
      embedding.zip(grads).foreach {
        case (e, g) => e.setStorage(new IntFloatSparseVectorStorage(e.dim().toInt, g))
      }
    } else {
      val grads = embedding.map(f => new Long2FloatOpenHashMap(f.getSize.toInt))
      for (i <- feats.indices)
        for (j <- 0 until embeddingDim)
          grads(j).addTo(feats(i), grad(i * embeddingDim + j))
      embedding.zip(grads).foreach {
        case (e, g) => e.setStorage(new LongFloatSparseVectorStorage(e.dim(), g))
      }
    }
  }

  def makeMatsGrad(grad: Array[Float], mats: IntFloatVector): Unit =
    mats.setStorage(new IntFloatDenseVectorStorage(grad))

  def makeMatsGrad(grad: Array[Float], mats: Vector): Unit =
    makeMatsGrad(grad, mats.asInstanceOf[IntFloatVector])

}
