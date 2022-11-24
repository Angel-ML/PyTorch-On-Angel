package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

import scala.collection.mutable

trait HasFeatureSplitIdxs extends Params {

  final val featureSplitIdxs = new Param[String](this, "featureSplitIdxs", "split idx for dense-high-sparse features, each type node separated by comma")

  final def getFeatureSplitIdxs: Map[String, Int] = {
    val dims = $(featureSplitIdxs).split(",")
    val dimsMap = new mutable.HashMap[String, Int]()
    dims.foreach{ p =>
      val kv = p.split(":")
      if (kv.length == 2) dimsMap.put(kv(0), kv(1).toInt)
    }
    dimsMap.toMap
  }

  final def getFeatureSplitIdxsByInt: Map[Int, Int] = {
    val dims = $(featureSplitIdxs).split(",")
    val dimsMap = new mutable.HashMap[Int, Int]()
    dims.foreach{ p =>
      val kv = p.split(":")
      if (kv.length == 2) dimsMap.put(kv(0).toInt, kv(1).toInt)
    }
    dimsMap.toMap
  }

  setDefault(featureSplitIdxs, "")

  final def setFeatureSplitIdxs(dims: String): this.type = set(featureSplitIdxs, dims)

}

