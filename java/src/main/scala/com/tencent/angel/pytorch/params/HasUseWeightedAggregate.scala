package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasUseWeightedAggregate extends Params{

  final val hasUseWeightedAggregate = new BooleanParam(this, "hasUseWeightedAggregate", "hasUseWeightedAggregate")

  setDefault(hasUseWeightedAggregate, false)

  final def getHasUseWeightedAggregate: Boolean = $(hasUseWeightedAggregate)

  final def setHasUseWeightedAggregate(value: Boolean): this.type = set(hasUseWeightedAggregate, value)

}
