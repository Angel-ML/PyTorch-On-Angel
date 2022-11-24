package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasLogStep extends Params{
  final val logStep = new IntParam(this, "logStep", "logStep")

  final def getLogStep: Int = $(logStep)

  setDefault(logStep, 10)

  final def setLogStep(value: Int): this.type = set(logStep, value)

}
