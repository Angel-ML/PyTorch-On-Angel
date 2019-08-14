package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasOutputModelPath extends Params {

  final val outputModelPath = new Param[String](this, "outputModelPath", "outputModelPath")

  final def getOutputModelPath: String = $(outputModelPath)

  setDefault(outputModelPath, "")

  final def setOutputModelPath(path: String): this.type = set(outputModelPath, path)

}
