package com.tencent.angel.pytorch.examples

import com.tencent.angel.pytorch.native.LibraryLoader
import org.apache.spark.{SparkConf, SparkContext}

object ClusterLoad {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    sc.makeRDD(Array(0, 1, 2, 3), 2).mapPartitions {
      case iterator =>
        LibraryLoader.load
        Iterator.single(1)
    }.count()

    Thread.sleep(100000)
  }

}
