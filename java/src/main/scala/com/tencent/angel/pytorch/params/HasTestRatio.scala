package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{FloatParam, Params}

trait HasTestRatio extends Params {

  final val testRatio = new FloatParam(this, "testRatio", "testRatio")

  final def getTestRatio: Float = $(testRatio)

  setDefault(testRatio, 0.5f)

  final def setTestRatio(ratio: Float): this.type = set(testRatio, ratio)
}
