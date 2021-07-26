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
package com.tencent.angel.pytorch.optim

import java.util.concurrent.Future
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.psf.update.base.{UpdateParam, VoidResult}
import com.tencent.angel.spark.ml.psf.optim.{AsyncAdamFunc, AsyncOptimParam}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}

class AsyncAdam(eta: Double, decay: Double = 0.0, gamma: Double = 0.99, beta: Double = 0.9)
  extends AsyncOptim(eta, decay) {

  override def getNumSlots(): Int = 3

  def getParam(matrixId: Int, grads: Array[Vector], offset: Int): UpdateParam = {
    new AsyncOptimParam(matrixId, grads, Array(getCurrentEta, gamma, beta), Array(offset, getNumSlots(), numSteps))
  }

  override def asyncUpdate(vector: PSVector, offset: Int, grad: Vector): Future[VoidResult] = {
    grad.setRowId(vector.id)
    val func = new AsyncAdamFunc(getParam(vector.poolId, Array(grad), offset))
    vector.psfUpdate(func)
  }

  override def asyncUpdate(matrix: PSMatrix, offset: Int, rowIds: Array[Int], grads: Array[Vector]): Future[VoidResult] = {
    assert(grads.length == rowIds.length)
    grads.zip(rowIds).foreach(f => f._1.setRowId(f._2))
    val func = new AsyncAdamFunc(getParam(matrix.id, grads, offset))
    matrix.psfUpdate(func)
  }

  override def asyncUpdate(matrix: PSMatrix, offset: Int, grads: Array[Vector]): Future[VoidResult] =
    asyncUpdate(matrix, offset, grads.indices.toArray, grads)

  override def getType: Int = 3

  override def toString: String =
    s"AsyncAdam ${super.toString}"

  override def getBeta: Float = beta.toFloat

  override def getGamma: Float = gamma.toFloat

  override def getEpsilon: Float = 1.0E-8f
}
