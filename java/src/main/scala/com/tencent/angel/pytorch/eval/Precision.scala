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
package com.tencent.angel.pytorch.eval

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

class Precision extends Evaluation {
  override def calculate(pairs: RDD[(Double, Double)]): Double = {
    pairs.persist(StorageLevel.MEMORY_ONLY)
    pairs.count()
    val partNum = pairs.getNumPartitions
    val tags = pairs.groupByKey(partNum)
      .persist(StorageLevel.MEMORY_ONLY)
    val classNum = tags.count().toInt
    val revTags = pairs.groupBy(f => f._2).repartition(partNum)
      .persist(StorageLevel.MEMORY_ONLY)
    revTags.count()
    pairs.unpersist(blocking = false)
    val prec = if (classNum == 2) {
      val tp = tags.map { info =>
        if (info._1 == 1) {
          info._2.map(o => if (o == 1) 1 else 0).sum
        } else 0
      }.sum
      val fp = revTags.map { info =>
        if (info._1 == 1) {
          info._2.map(o => if (o._1 == 0) 1 else 0).sum
        } else 0
      }.sum
      if (tp + fp != 0) 1.0 * tp / (tp + fp) else 0.0
    } else {
      val tps = tags.map { info =>
        val tp = info._2.map(o => if (o == info._1) 1 else 0).sum
        (info._1, tp)
      }
      val fps = revTags.map { info =>
        val fp = info._2.map(t => if (t._1 != info._1) 1 else 0).sum
        (info._1, fp)
      }
      // (tar, (tp, Option(fp)))
      tps.leftOuterJoin(fps).map { it =>
        val tfp = it._2._2
        tfp match {
          case Some(tfp) =>
            val tmp = it._2._1 + tfp
            if (tmp == 0) 0.0 else 1.0 * it._2._1 / tmp
          case None =>
            val tmp = it._2._1
            if (tmp == 0) 0.0 else 1.0 * it._2._1 / tmp
        }
      }.sum / classNum
    }
    tags.unpersist(blocking = false)
    revTags.unpersist(blocking = false)
    prec
  }

  def calculate1(pairs: RDD[(Double, Double)]): Double = {
    pairs.cache()
    val classNum = pairs.groupBy(f => f._1).count().toInt
    if (classNum == 2) {
      val tp = pairs.filter(f => f._1 == f._2 && f._1 == 1).count()
      val fp = pairs.filter(f => f._1 == 0 && f._2 == 1).count()
      tp * 1.0 / (tp + fp)
    } else {
      (0 until classNum).map { i =>
        val tp = pairs.filter(f => f._1 == f._2 && f._1 == i).count()
        val fp = pairs.filter(f => f._1 != i && f._2 == i).count()
        tp * 1.0 / (tp + fp)
      }.sum / classNum
    }
  }
}
