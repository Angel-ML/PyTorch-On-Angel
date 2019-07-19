package com.tencent.angel.pytorch.model

import com.tencent.angel.ml.matrix.RowType

import scala.collection.mutable

class TorchParams extends Serializable {

  var modelType: String = ""
  var dim: Long = -1
  var embeddingDim: Int = -1
  var numFields: Int = -1
  var rowType: RowType = RowType.T_FLOAT_DENSE
  var matSizes: Array[Int] = Array()
  val map = new mutable.HashMap[String, String]()

}
