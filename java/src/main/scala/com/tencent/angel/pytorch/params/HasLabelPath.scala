package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasLabelPath extends Params {

  final val labelPath = new Param[String](this, "labelPath", "labelPath")

  final def getLabelPath: String = $(labelPath)

  setDefault(labelPath, "")

  final def setLabelPath(path: String): this.type = set(labelPath, path)

}
