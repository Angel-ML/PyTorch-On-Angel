package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{IntParam, Params}
trait HasWindowSize extends Params {

  final val windowSize = new IntParam(this, "windowSize", "windowSize")

  final def getWindowSize: Int = $(windowSize)

  setDefault(windowSize, 10)

  final def setWindowSize(value: Int): this.type = set(windowSize, value)

}