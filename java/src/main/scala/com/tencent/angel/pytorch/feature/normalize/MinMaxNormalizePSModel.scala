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
import com.tencent.angel.ml.matrix.psf.update.enhance.complex.{Max, Min}
import com.tencent.angel.ml.matrix.psf.update.update.IncrementRowsParam
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.models.PSVector
import com.tencent.angel.spark.models.impl.PSVectorImpl

private[normalize]
class MinMaxNormalizePSModel(max: PSVector, min: PSVector) extends Serializable {

  def init(): Unit = {
    max.fill(Float.MinValue)
    min.fill(Float.MaxValue)
  }

  def readMax(): IntFloatVector =
    max.pull().asInstanceOf[IntFloatVector]

  def readMin(): IntFloatVector =
    min.pull().asInstanceOf[IntFloatVector]

  def updateMax(update: Vector): Unit = {
    update.setRowId(max.id)
    val func = new Max(new IncrementRowsParam(max.poolId, Array(update)))
    max.psfUpdate(func).get()
  }

  def updateMin(update: Vector): Unit = {
    update.setRowId(min.id)
    val func = new Min(new IncrementRowsParam(min.poolId, Array(update)))
    min.psfUpdate(func).get()
  }

}

private[normalize]
object MinMaxNormalizePSModel {
  def apply(minId: Long, maxId: Long): MinMaxNormalizePSModel = {
    val feature = new MatrixContext("feature", 2, minId, maxId)
    feature.setValidIndexNum(-1)
    feature.setRowType(RowType.T_FLOAT_DENSE)

    val featureId = PSContext.instance().createMatrix(feature).getId

    new MinMaxNormalizePSModel(
      new PSVectorImpl(featureId, 0, maxId, feature.getRowType),
      new PSVectorImpl(featureId, 1, maxId, feature.getRowType))
  }
}
