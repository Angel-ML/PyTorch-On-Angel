package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasDataFormat extends Params {

  final val dataFormat = new Param[String](this, "dataFormat", "dataFormat")

  final def getDataFormat: String = $(dataFormat)

  setDefault(dataFormat, "sparse")

  final def setDataFormat(format: String): this.type = set(dataFormat, format)
}
