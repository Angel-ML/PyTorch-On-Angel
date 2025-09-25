package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasOutputEmbeddingWithType extends Params {
  val outputEmbeddingByNodeType = new BooleanParam(this, "outputEmbeddingByNodeType", "outputEmbeddingByNodeType")

  setDefault(outputEmbeddingByNodeType, false)

  final def setOutputEmbeddingByNodeType(flag: Boolean): this.type = set(outputEmbeddingByNodeType, flag)

  final def getOutputEmbeddingByNodeType: Boolean = $(outputEmbeddingByNodeType)


  val outputEmbeddingByEdgeType = new BooleanParam(this, "outputEmbeddingByEdgeType", "outputEmbeddingByEdgeType")

  setDefault(outputEmbeddingByEdgeType, false)

  final def setOutputEmbeddingByEdgeType(flag: Boolean): this.type = set(outputEmbeddingByEdgeType, flag)

  final def getOutputEmbeddingByEdgeType: Boolean = $(outputEmbeddingByEdgeType)
}
