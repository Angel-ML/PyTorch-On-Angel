package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasOutputFeaturePath extends Params {

  final val outputFeaturePath = new Param[String](this, "outputFeaturePath", "outputFeaturePath")

  final def getOutputFeaturePath: String = $(outputFeaturePath)

  setDefault(outputFeaturePath, "")

  final def setOutputFeaturePath(path: String): this.type = set(outputFeaturePath, path)

}
