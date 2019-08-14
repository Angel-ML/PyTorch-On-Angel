package com.tencent.angel.pytorch.graph.subgraph

import it.unimi.dsi.fastutil.longs.LongArrayList

import scala.collection.mutable.ArrayBuffer

private[subgraph]
class FeaturePartition(index: Int,
                       keys: Array[Long],
                       features: Array[String]) extends Serializable {

  def sample(model: SubGraphPSModel): Iterator[(Long, String)] = {
    val flags = model.readNodes(keys.clone())
    val results = new ArrayBuffer[(Long, String)]()
    for (idx <- keys.indices) {
      if (flags.get(keys(idx)) > 0) {
        results.append((keys(idx), features(idx)))
      }
    }
    results.iterator
  }
}

private[subgraph]
object FeaturePartition {
//  def apply(index: Int, iterator: Iterator[(Long, String)]): FeaturePartition = {
//    val fs = iterator.toArray
//    val keys = fs.map(f => f._1)
//    val features = fs.map(f => f._2)
//    new FeaturePartition(index, keys, features)
//  }

  def apply(index: Int, iterator: Iterator[String]): FeaturePartition = {
    val keys = new LongArrayList()
    val features = new ArrayBuffer[String]()
    while (iterator.hasNext) {
      val line = iterator.next()
      val parts = line.stripLineEnd.split(" ")

      keys.add(parts(0).toLong)
      features.append(parts.tail.mkString(" "))
    }
    new FeaturePartition(index, keys.toLongArray, features.toArray)
  }
}
