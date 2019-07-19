package com.tencent.angel.pytorch.params

import com.tencent.angel.pytorch.optim.{AsyncOptim, OptimUtils}
import org.apache.spark.ml.param.{Param, Params}

trait HasOptimizer extends Params {

  final val optimizer = new Param[String](this, "optimizer", "optimizer")

  final def getOptimizer: AsyncOptim = OptimUtils.apply($(optimizer), $(stepSize))

  setDefault(optimizer, "SGD")

  final def setOptimizer(value: String): this.type = set(optimizer, value)

  final val stepSize = new Param[Double](this, "stepSize", "stepSize")

  final def getStepSize: Double = $(stepSize)

  setDefault(stepSize, 0.01)

  final def setStepSize(value: Double): this.type = set(stepSize, value)
}
