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
package com.tencent.angel.pytorch.feature.normalize

import com.tencent.angel.ml.math2.vector.{IntFloatVector, Vector}
import com.tencent.angel.ml.matrix.psf.update.enhance.complex.Add
import com.tencent.angel.ml.matrix.psf.update.update.IncrementRowsParam
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.models.PSVector
import com.tencent.angel.spark.models.impl.PSVectorImpl
import com.tencent.angel.spark.util.VectorUtils

private[normalize]
class MeanVarNormalizePSModel(sum: PSVector,
                              count: PSVector,
                              variance: PSVector) extends Serializable {
  def addSum(update: Vector): Unit = {
    update.setRowId(sum.id)
    val func = new Add(new IncrementRowsParam(sum.poolId, Array(update)))
    sum.psfUpdate(func).get()
  }

  def addVar(update: Vector): Unit = {
    update.setRowId(variance.id)
    val func = new Add(new IncrementRowsParam(variance.poolId, Array(update)))
    variance.psfUpdate(func).get()
  }

  def calculateMean(count: Long): Unit =
    VectorUtils.idiv(sum, count.toDouble)

  def calculateVar(count: Long): Unit = {
    VectorUtils.idiv(variance, count.toDouble)
    VectorUtils.isqrt(variance)
  }

  def readVariance(): IntFloatVector =
    variance.pull().asInstanceOf[IntFloatVector]

  def readMean(): IntFloatVector =
    sum.pull().asInstanceOf[IntFloatVector]

  def readSum(): IntFloatVector =
    sum.pull().asInstanceOf[IntFloatVector]

  def addSumCount(sumUpdate: Vector, countUpdate: Vector): Unit = {
    sumUpdate.setRowId(sum.id)
    countUpdate.setRowId(count.id)
    val func = new Add(new IncrementRowsParam(sum.poolId, Array(sumUpdate, countUpdate)))
    sum.psfUpdate(func).get()
  }
}

private[normalize]
object MeanVarNormalizePSModel {
  def apply(minId: Long, maxId: Long): MeanVarNormalizePSModel = {
    val matrix = new MatrixContext("matrix", 3, minId, maxId)
    matrix.setValidIndexNum(-1)
    matrix.setRowType(RowType.T_FLOAT_DENSE)

    val matrixId = PSContext.instance().createMatrix(matrix).getId

    new MeanVarNormalizePSModel(
      new PSVectorImpl(matrixId, 0, maxId, matrix.getRowType),
      new PSVectorImpl(matrixId, 1, maxId, matrix.getRowType),
      new PSVectorImpl(matrixId, 2, maxId, matrix.getRowType))
  }
}
