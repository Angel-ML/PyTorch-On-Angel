package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasFilterSameNode extends Params {

  final val filterSameNode = new BooleanParam(this, "filterSameNode", "filterSameNode")

  final def getFilterSameNode: Boolean = $(filterSameNode)

  setDefault(filterSameNode, true)

  final def setFilterSameNode(filter: Boolean): this.type = set(filterSameNode, filter)

}
