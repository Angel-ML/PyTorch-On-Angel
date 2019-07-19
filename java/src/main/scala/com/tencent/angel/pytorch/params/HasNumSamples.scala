package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasNumSamples extends Params {
  final val numSamples = new IntParam(this, "numSamples", "numSamples")

  final def getNumSamples: Int = $(numSamples)

  setDefault(numSamples, 5)

  final def setNumSamples(value: Int): this.type = set(numSamples, value)
}
