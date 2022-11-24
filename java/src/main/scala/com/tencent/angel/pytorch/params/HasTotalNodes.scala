package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasTotalNodes extends Params {

  final val totalNodes = new Param[String](this, "totalNodes", "totalNodes separated by comma")

  final val keyNode = new Param[String](this, "keyNode", "start from key node to sample neighbors")

  final def getTotalNodes: String = $(totalNodes)

  setDefault(totalNodes, "")

  final def setTotalNodes(nodes: String): this.type = set(totalNodes, nodes)

  final def getKeyNode: String = $(keyNode)

  setDefault(keyNode, "u")

  final def setKeyNode(node: String): this.type = set(keyNode, node)

}
