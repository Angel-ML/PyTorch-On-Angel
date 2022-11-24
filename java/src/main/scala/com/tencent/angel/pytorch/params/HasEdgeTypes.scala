package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param,Params}

trait HasEdgeTypes extends Params {

  final val edgeTypes = new Param[String](this, "edgeTypes", "edgeType separated by comma")

  final def getEdgeTypes: String = $(edgeTypes)

  setDefault(edgeTypes, "")

  final def setEdgeTypes(types: String): this.type = set(edgeTypes, types)

}
