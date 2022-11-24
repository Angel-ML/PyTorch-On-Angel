package com.tencent.angel.pytorch.params


import org.apache.spark.ml.param.{Param, Params}

trait HasContextEmbedPath extends Params {

  final val contextEmbedPath = new Param[String](this, "contextEmbedPath", "inputContextEmbedPath")

  final def getContextEmbedPath: String = $(contextEmbedPath)

  setDefault(contextEmbedPath, "")

  final def setContextEmbedPath(path: String): this.type = set(contextEmbedPath, path)

}
