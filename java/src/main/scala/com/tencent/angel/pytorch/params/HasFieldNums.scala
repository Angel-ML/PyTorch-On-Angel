package com.tencent.angel.pytorch.params

import org.apache.spark.ml.param.{Param, Params}

import scala.collection.mutable

trait HasFieldNums extends Params {

  final val fieldNums = new Param[String](this, "fieldNums", "fieldNums for each type node separated by comma")

  final def getFieldNums: Map[String, Int] = {
    val nums = $(fieldNums).split(",")
    val fieldNumsMap = new mutable.HashMap[String, Int]()
    nums.foreach{ p =>
      val kv = p.split(":")
      if (kv.length == 2)
        fieldNumsMap.put(kv(0), kv(1).toInt)
    }
    fieldNumsMap.toMap
  }

  final def getFieldNumsByInt: Map[Int, Int] = {
    val nums = $(fieldNums).split(",")
    val fieldNumsMap = new mutable.HashMap[Int, Int]()
    nums.foreach{ p =>
      val kv = p.split(":")
      if (kv.length == 2)
        fieldNumsMap.put(kv(0).toInt, kv(1).toInt)
    }
    fieldNumsMap.toMap
  }

  setDefault(fieldNums, "")

  final def setFieldNums(fields: String): this.type = set(fieldNums, fields)

}
