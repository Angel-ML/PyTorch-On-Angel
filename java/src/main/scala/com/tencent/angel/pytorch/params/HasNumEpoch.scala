package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasNumEpoch extends Params {

  final val numEpoch = new IntParam(this, "numEpoch", "numEpoch")

  final def getNumEpoch: Int = $(numEpoch)

  setDefault(numEpoch, 10)

  final def setNumEpoch(value: Int): this.type = set(numEpoch, value)

}
