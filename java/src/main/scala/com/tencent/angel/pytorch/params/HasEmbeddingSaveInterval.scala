package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasEmbeddingSaveInterval extends Params {

  final val saveEmbeddingInterval = new IntParam(this, "saveEmbeddingInterval", "saveEmbeddingInterval")

  setDefault(saveEmbeddingInterval, Int.MaxValue)

  final def getSaveEmbeddingInterval: Int = $(saveEmbeddingInterval)

  final def setSaveEmbeddingInterval(value: Int): this.type = set(saveEmbeddingInterval, value)

}
