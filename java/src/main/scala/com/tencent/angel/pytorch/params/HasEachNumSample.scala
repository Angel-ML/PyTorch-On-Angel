package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

import scala.collection.mutable

trait HasEachNumSample extends Params{
  /**
   * Param for buffer size.
   *
   * @group param
   */
  final val eachNumSample = new Param[String](this, "eachNumSample", "eachNumSample")

  /** @group getParam */
  final def getEachNumSample: Map[Int, Int] = {
    val num = $(eachNumSample).split(",")
    val epochNumMap = new mutable.HashMap[Int, Int]()
    num.foreach{ p =>
      val kv = p.split(":")
      if (kv.length == 2) epochNumMap.put(kv(0).toInt, kv(1).toInt)
    }
    epochNumMap.toMap
  }

  setDefault(eachNumSample, "")

  /** @group setParam */
  final def setEachNumSample(num: String): this.type = set(eachNumSample, num)
}
