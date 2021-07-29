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

class AUC extends Evaluation {

  override
  def calculate(pairs: RDD[(Double, Double)]): Double = {
    // sort by predict
    val sorted = pairs.sortBy(f => f._2)
    sorted.cache()
    val numTotal = sorted.count()
    val numPositive = sorted.filter(f => f._1 > 0).count()
    val numNegetive = numTotal - numPositive
    // calculate the summation of ranks for positive samples
    val sumRanks = sorted.zipWithIndex().filter(f => f._1._1.toInt == 1).map(f => f._2 + 1).reduce(_ + _)
    val auc = sumRanks * 1.0 / numPositive / numNegetive - (numPositive + 1.0) / 2.0 / numNegetive
    sorted.unpersist()
    auc
  }
}
