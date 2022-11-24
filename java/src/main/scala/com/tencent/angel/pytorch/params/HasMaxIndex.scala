package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasMaxIndex extends Params{

  final val maxIndex = new IntParam(this, "maxIndex", "maxIndex")

  final def getMaxIndex: Int = $(maxIndex)

  setDefault(maxIndex, 10)

  final def setMaxIndex(value: Int): this.type = set(maxIndex, value)

}
