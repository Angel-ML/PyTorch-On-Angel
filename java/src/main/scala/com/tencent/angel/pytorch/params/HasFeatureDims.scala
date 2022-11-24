package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

import scala.collection.mutable

trait HasFeatureDims extends Params {

  final val featureDims = new Param[String](this, "featureDims", "featureDims for each type node separated by comma")

  final def getFeatureDims: Map[String, Long] = {
    val dims = $(featureDims).split(",")
    val dimsMap = new mutable.HashMap[String, Long]()
    dims.foreach{ p =>
      val kv = p.split(":")
      if (kv.length == 2) dimsMap.put(kv(0), kv(1).toInt)
    }
    dimsMap.toMap
  }

  final def getFeatureDimsByInt: Map[Int, Int] = {
    val dims = $(featureDims).split(",")
    val dimsMap = new mutable.HashMap[Int, Int]()
    dims.foreach{ p =>
      val kv = p.split(":")
      if (kv.length == 2) dimsMap.put(kv(0).toInt, kv(1).toInt)
    }
    dimsMap.toMap
  }

  setDefault(featureDims, "")

  final def setFeatureDims(dims: String): this.type = set(featureDims, dims)

}
