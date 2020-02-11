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
package com.tencent.angel.pytorch.params

import com.tencent.angel.pytorch.optim.{AsyncOptim, OptimUtils}
import org.apache.spark.ml.param.{Param, Params}

trait HasOptimizer extends Params {

  final val optimizer = new Param[String](this, "optimizer", "optimizer")

  final def getOptimizer: AsyncOptim = OptimUtils.apply($(optimizer), $(stepSize), $(decay))

  setDefault(optimizer, "SGD")

  final def setOptimizer(value: String): this.type = set(optimizer, value)

  final val stepSize = new Param[Double](this, "stepSize", "stepSize")

  final def getStepSize: Double = $(stepSize)

  setDefault(stepSize, 0.01)

  final def setStepSize(value: Double): this.type = set(stepSize, value)

  final val decay = new Param[Double](this, "decay", "decay")

  final def getDecay: Double = $(decay)

  setDefault(decay, 0.0)

  final def setDecay(value: Double): this.type = set(decay, value)
}
