package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasNumBatchInit extends Params {

  final val numBatchInit = new IntParam(this, "numBatchInit", "numBatchInit")

  final def getNumBatchInit: Int = $(numBatchInit)

  setDefault(numBatchInit, 4)

  final def setNumBatchInit(num: Int): this.type = set(numBatchInit, num)

}
