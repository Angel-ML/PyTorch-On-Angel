package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasContextDim extends Params{

  final val contextDim = new IntParam(this, "contextDim", "contextDim")

  final def getContextDim: Int = $(contextDim)

  setDefault(contextDim, 0)

  final def setContextDim(dim: Int): this.type = set(contextDim, dim)

}
