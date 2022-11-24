package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

import scala.collection.mutable

trait HasEmbedDims extends Params {

  final val embedDims = new Param[String](this, "embedDims", "embedDims for each type node separated by comma")

  final def getEmbedDims: Map[String, Int] = {
    val dims = $(embedDims).split(",")
    val dimsMap = new mutable.HashMap[String, Int]()
    dims.foreach{ p =>
      val kv = p.split(":")
      if (kv.length == 2)
        dimsMap.put(kv(0), kv(1).toInt)
    }
    dimsMap.toMap
  }

  final def getEmbedDimsByInt: Map[Int, Int] = {
    val dims = $(embedDims).split(",")
    val dimsMap = new mutable.HashMap[Int, Int]()
    dims.foreach{ p =>
      val kv = p.split(":")
      if (kv.length == 2)
        dimsMap.put(kv(0).toInt, kv(1).toInt)
    }
    dimsMap.toMap
  }

  setDefault(embedDims, "")

  final def setEmbedDims(dims: String): this.type = set(embedDims, dims)

}
