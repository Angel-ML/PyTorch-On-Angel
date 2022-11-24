package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param,Params}

trait HasNodeTypes extends Params {

  final val nodeTypes = new Param[String](this, "nodeTypes", "nodeType separated by comma")

  final def getNodeTypes: String = $(nodeTypes)

  setDefault(nodeTypes, "")

  final def setNodeTypes(types: String): this.type = set(nodeTypes, types)

}
