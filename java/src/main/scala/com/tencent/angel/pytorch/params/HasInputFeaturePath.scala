package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params, StringArrayParam}

trait HasInputFeaturePath extends Params {

  final val inputFeaturePath = new Param[String](this, "inputFeaturePath", "inputFeaturePath")

  final def getInputFeaturePath: String = $(inputFeaturePath)

  setDefault(inputFeaturePath, "")

  final def setInputFeaturePath(path: String): this.type = set(inputFeaturePath, path)

}
