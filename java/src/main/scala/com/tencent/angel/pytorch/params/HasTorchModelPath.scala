package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasTorchModelPath extends Params {
  final val torchModelPath = new Param[String](this, "torchModelPath", "torchModelPath")

  final def getTorchModelPath: String = $(torchModelPath)

  setDefault(torchModelPath, "")

  final def setTorchModelPath(path: String): this.type = set(torchModelPath, path)
}
