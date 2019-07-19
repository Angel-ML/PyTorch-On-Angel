package com.tencent.angel.pytorch.optim

import java.util.concurrent.Future

import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.psf.update.base.VoidResult
import com.tencent.angel.spark.models.{PSMatrix, PSVector}

abstract class AsyncOptim extends Serializable {

  def update(vector: PSVector, offset: Int, grad: Vector) = {
    asycUpdate(vector, offset, grad).get
  }

  def update(matrix: PSMatrix, offset: Int, rowIds: Array[Int], grads: Array[Vector]) = {
    asycUpdate(matrix, offset, rowIds, grads).get
  }

  def update(matrix: PSMatrix, offset: Int, grads: Array[Vector]) = {
    asycUpdate(matrix, offset, grads).get
  }

  def asycUpdate(vector: PSVector, offset: Int, grad: Vector): Future[VoidResult]

  def asycUpdate(matrix: PSMatrix, offset: Int, rowIds: Array[Int], grads: Array[Vector]): Future[VoidResult]

  def asycUpdate(matrix: PSMatrix, offset: Int, grads: Array[Vector]): Future[VoidResult]

  def getNumSlots(): Int
}
