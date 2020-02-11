/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
package com.tencent.angel.pytorch.partition

import com.tencent.angel.ml.matrix.{MatrixContext, PartContext}
import org.apache.spark.rdd.RDD

object LoadBalancePartitioner {

  def getBuckets(data: RDD[Long], bits: Int): Array[(Long, Long)] = {
    val buckets = data.map(f => f / (1L << bits))
      .map(f => (f, 1L)).reduceByKey(_ + _, 100).sortBy(_._1).collect()
    return buckets
  }

  def partition(buckets: Array[(Long, Long)], ctx: MatrixContext, bits: Int, numPartitions: Int): Unit = {
    println(s"bucket.size=${buckets.size}")
    val sorted = buckets.sortBy(f => f._1)
    val sum = sorted.map(f => f._2).sum
    val per = sum / numPartitions

    var start = ctx.getIndexStart
    val end = ctx.getIndexEnd
    val rowNum = ctx.getRowNum

    var current = 0L
    val size = sorted.size
    val limit = ((end.toDouble - start.toDouble) / numPartitions).toLong * 4
    println(s"range limit=$limit persize=$per sum=$sum numPartitions=$numPartitions")
    for (i <- 0 until size) {
      // keep each partition similar load and limit the range of each partition'
      val range = (sorted(i)._1 << bits) - start
      if (current > per
        || ((range > limit) && (current > per / 2))
        || (range > (2 * limit))) {
        val part = new PartContext(0, rowNum, start, sorted(i)._1 << bits, 0)
        println(s"part=$part load=$current range=${part.getEndCol - part.getStartCol}")
        ctx.addPart(part)
        start = sorted(i)._1 << bits
        current = 0L
      }
      current += sorted(i)._2
    }

    val part = new PartContext(0, rowNum, start, end, 0)
    ctx.addPart(part)
    println(s"part=${part} load=${current} range=${end - start}")
    println(s"split matrix ${ctx.getName} into ${ctx.getParts.size()} partitions")
  }

  def partition(index: RDD[Long], maxId: Long, psPartitionNum: Int, ctx: MatrixContext, percent: Float = 0.7f): Unit = {
    var p = percent
    var count = 2
    while (count > 0) {
      val bits = (numBits(maxId) * p).toInt
      println(s"bits used for load balance partition is $bits")
      val buckets = getBuckets(index, bits)
      val sum = buckets.map(f => f._2).sum
      val max = buckets.map(f => f._2).max
      val size = buckets.size
      println(s"max=$max sum=$sum size=$size")
      if (count == 1 || max.toDouble / sum < 0.1) {
        count = 0
        partition(buckets, ctx, bits, psPartitionNum)
      } else {
        count -= 1
        p -= 0.05f
      }
    }
  }

  def numBits(maxId: Long): Int = {
    var num = 0
    var value = maxId
    while (value > 0) {
      value >>= 1
      num += 1
    }
    num
  }
}