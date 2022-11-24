package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasMetaPaths extends Params {

  final val metaPaths = new Param[String](this, "metaPaths", "metaPaths for sampling separated by comma")

  final def getMetaPaths: String = $(metaPaths)

  setDefault(metaPaths, "")

  final def setMetaPaths(path: String): this.type = set(metaPaths, path)

}
