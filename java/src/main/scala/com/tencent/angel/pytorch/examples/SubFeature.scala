package com.tencent.angel.pytorch.examples

import com.tencent.angel.spark.ml.core.ArgsUtil
import it.unimi.dsi.fastutil.ints.IntOpenHashSet
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

object SubFeature {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val featureInput = params.getOrElse("featureInput", "")
    val labelInput = params.getOrElse("labelInput", "")
    val output = params.getOrElse("output", "")

    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val features = sc.textFile(featureInput)
    val labels = sc.textFile(labelInput).map(f => f.stripLineEnd.split(" ")(0).toInt).collect()

    def extract(iterator: Iterator[String]): Iterator[String] = {
      val set = new IntOpenHashSet()
      labels.foreach(f => set.add(f))
      val results = new ArrayBuffer[String]()
      while (iterator.hasNext) {
        val line = iterator.next()
        val parts = line.stripLineEnd.split(" ")
        val key = parts(0).toInt
        if (labels.contains(key))
          results.append(line)
      }
      results.iterator
    }

    features.mapPartitions(extract).saveAsTextFile(output)

  }

}
