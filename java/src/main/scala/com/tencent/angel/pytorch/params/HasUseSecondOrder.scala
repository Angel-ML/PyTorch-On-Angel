package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasUseSecondOrder extends Params {

  final val useSecondOrder = new BooleanParam(this, "useSecondOrder", "useSecondOrder")

  final def getUseSecondOrder: Boolean = $(useSecondOrder)

  setDefault(useSecondOrder, true)

  final def setUseSecondOrder(flag: Boolean): this.type = set(useSecondOrder, flag)

}
