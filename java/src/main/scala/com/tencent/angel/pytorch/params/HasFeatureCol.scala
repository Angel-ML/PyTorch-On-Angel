package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasFeatureCol extends Params {

  final val featureCol = new Param[String](this, "feature", "feature")

  final def getFeatureCol: String = $(featureCol)

  setDefault(featureCol, "feature")

  final def setFeatureCol(name: String): this.type = set(featureCol, name)

}
