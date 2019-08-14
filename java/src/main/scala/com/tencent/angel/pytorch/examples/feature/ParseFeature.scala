package com.tencent.angel.pytorch.examples.feature

import com.tencent.angel.spark.ml.core.ArgsUtil
import org.apache.spark.{SparkConf, SparkContext}

object ParseFeature {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "")
    val output = params.getOrElse("output", "")

    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    sc.textFile(input).map {
      case line =>
        val parts = line.stripLineEnd.split(" ")
        val (nodeId, fs) = (parts.head, parts.tail)
        (nodeId.toLong, fs)
    }.map(f => s"${f._1}\t${f._2.mkString(" ")}")
      .saveAsTextFile(output)
  }

}
