package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

trait HasSchema extends Params {
  final val schema = new Param[String](this, "schema", "schema separated by comma")

  final def getSchema: String = $(schema)

  setDefault(schema, "")

  final def setSchema(values: String): this.type = set(schema, values)

}
