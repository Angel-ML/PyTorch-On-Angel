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

class F1Score extends Evaluation {

  override def calculate(pairs: RDD[(Double, Double)]): Double = {
    pairs.cache()
    val classNum = pairs.groupBy(f => f._1).count().toInt
    if (classNum == 2) {
      val tp = pairs.filter(f => f._1 == f._2 && f._1 == 1).count()
      val fp = pairs.filter(f => f._1 == 0 && f._2 == 1).count()
      val fn = pairs.filter(f => f._1 == 1 && f._2 == 0).count()
      2.0 * tp / (2 * tp + fp + fn)
    } else {
      (0 until classNum).map { i =>
        val tp = pairs.filter(f => f._1 == f._2 && f._1 == i).count()
        val fp = pairs.filter(f => f._1 != i && f._2 == i).count()
        val fn = pairs.filter(f => f._1 == i && f._2 != i).count()
        2.0 * tp / (2 * tp + fp + fn)
      }.sum / classNum
    }
  }
}
