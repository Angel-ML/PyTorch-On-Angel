package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasNegSampleByNodeType extends Params {

  final val negSampleByNodeType = new BooleanParam(this, "negSampleByNodeType", "whether to sample negatives by mode types")

  final def getNegSampleByNodeType: Boolean = $(negSampleByNodeType)

  setDefault(negSampleByNodeType, false)

  final def setNegSampleByNodeType(flag: Boolean): this.type = set(negSampleByNodeType, flag)
}
