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
import com.tencent.angel.spark.models.{PSMatrix, PSVector}

abstract class AsyncOptim(eta: Double, decay: Double) extends Serializable {

  protected var numSteps: Int = 1

  def update(vector: PSVector, offset: Int, grad: Vector) = {
    asyncUpdate(vector, offset, grad).get
  }

  def update(matrix: PSMatrix, offset: Int, rowIds: Array[Int], grads: Array[Vector]) = {
    asyncUpdate(matrix, offset, rowIds, grads).get
  }

  def update(matrix: PSMatrix, offset: Int, grads: Array[Vector]) = {
    asyncUpdate(matrix, offset, grads).get
  }

  def asyncUpdate(vector: PSVector, offset: Int, grad: Vector): Future[VoidResult]

  def asyncUpdate(matrix: PSMatrix, offset: Int, rowIds: Array[Int], grads: Array[Vector]): Future[VoidResult]

  def asyncUpdate(matrix: PSMatrix, offset: Int, grads: Array[Vector]): Future[VoidResult]

  def getNumSlots(): Int

  def step(num: Int): Unit = numSteps += num

  def getCurrentStep: Int = numSteps

  def getCurrentEta: Double = eta / (1 + (numSteps - 1) * decay)

  override def toString: String = {
    s"eta=$eta decay=$decay"
  }
}