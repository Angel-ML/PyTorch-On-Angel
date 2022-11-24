package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

import scala.collection.mutable

trait HasNodeNumSamples extends Params {

  final val nodeNumSamples = new Param[String](this, "nodeNumSamples", "numSamples for each type node separated by comma")

  final def getNodeNumSamples: Map[String, Int] = {
    val samples = $(nodeNumSamples).split(",")
    val samplesMap = new mutable.HashMap[String, Int]()
    samples.foreach{ p =>
      val kv = p.split(":")
      samplesMap.put(kv(0), kv(1).toInt)
    }
    samplesMap.toMap
  }

  setDefault(nodeNumSamples, "")

  final def setNodeNumSamples(samples: String): this.type = set(nodeNumSamples, samples)

}
