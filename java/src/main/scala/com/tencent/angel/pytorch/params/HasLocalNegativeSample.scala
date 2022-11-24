package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasLocalNegativeSample extends Params {

  final val localNegativeSample = new BooleanParam(this, "localNegativeSample", "localNegativeSample")

  final def getLocalNegativeSample: Boolean = $(localNegativeSample)

  setDefault(localNegativeSample, true)

  final def setLocalNegativeSample(value: Boolean): this.type = set(localNegativeSample, value)
}