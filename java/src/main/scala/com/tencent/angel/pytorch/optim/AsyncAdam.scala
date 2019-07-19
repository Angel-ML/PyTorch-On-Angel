package com.tencent.angel.pytorch.optim

import java.util.concurrent.Future

import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.psf.update.base.{UpdateParam, VoidResult}
import com.tencent.angel.spark.ml.psf.optim.{AsyncAdamFunc, AsyncOptimParam}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}

class AsyncAdam(eta: Double, gamma: Double = 0.99, beta: Double = 0.9) extends AsyncOptim {

  var numUpdates = 0

  override def getNumSlots(): Int = 3

  def getParam(matrixId: Int, grads: Array[Vector], offset: Int): UpdateParam = {
    numUpdates += 1
    new AsyncOptimParam(matrixId, grads, Array(eta, gamma, beta), Array(offset, getNumSlots(), numUpdates))
  }

  override def asycUpdate(vector: PSVector, offset: Int, grad: Vector): Future[VoidResult] = {
    grad.setRowId(vector.id)
    val func = new AsyncAdamFunc(getParam(vector.poolId, Array(grad), offset))
    vector.psfUpdate(func)
  }

  override def asycUpdate(matrix: PSMatrix, offset: Int, rowIds: Array[Int], grads: Array[Vector]): Future[VoidResult] = {
    assert(grads.length == rowIds.length)
    grads.zip(rowIds).foreach(f => f._1.setRowId(f._2))
    val func = new AsyncAdamFunc(getParam(matrix.id, grads, offset))
    matrix.psfUpdate(func)
  }

  override def asycUpdate(matrix: PSMatrix, offset: Int, grads: Array[Vector]): Future[VoidResult] =
    asycUpdate(matrix, offset, (0 until grads.length).toArray, grads)
}
