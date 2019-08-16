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
import com.tencent.angel.ml.matrix.psf.update.base.VoidResult
import com.tencent.angel.spark.ml.psf.optim.{AsyncOptimParam, AsyncSGDFunc}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}

class AsyncSGD(stepSize: Double) extends AsyncOptim {
  override
  def getNumSlots(): Int = 1

  override def asycUpdate(vector: PSVector, offset: Int, grad: Vector): Future[VoidResult] = {
    grad.setRowId(vector.id)
    val param = new AsyncOptimParam(vector.poolId, Array(grad), Array(stepSize), Array(offset, getNumSlots()))
    val func = new AsyncSGDFunc(param)
    vector.psfUpdate(func)
  }

  override def asycUpdate(matrix: PSMatrix, offset: Int, rowIds: Array[Int], grads: Array[Vector]): Future[VoidResult] = {
    grads.zip(rowIds).map(f => f._1.setRowId(f._2))
    val param = new AsyncOptimParam(matrix.id, grads, Array(stepSize), Array(offset, getNumSlots()))
    val func = new AsyncSGDFunc(param)
    matrix.psfUpdate(func)
  }

  override def asycUpdate(matrix: PSMatrix, offset: Int, grads: Array[Vector]): Future[VoidResult] = {
    asycUpdate(matrix, offset, (0 until grads.length).toArray, grads)
  }
}
