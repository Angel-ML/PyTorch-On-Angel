package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasFeatureDim extends Params {

  final val featureDim = new IntParam(this, "featureDim", "featureDim")

  final def getFeatureDim: Int = $(featureDim)

  setDefault(featureDim, 0)

  final def setFeatureDim(dim: Int): this.type = set(featureDim, dim)

}
