package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasInitMethod extends Params {

  final val initMethod = new Param[String](this, "initMethod", "initMethod")

  final def getInitMethod: String = $(initMethod)

  setDefault(initMethod, "xavierUniform")

  final def setInitMethod(value: String): this.type = set(initMethod, value)
}
