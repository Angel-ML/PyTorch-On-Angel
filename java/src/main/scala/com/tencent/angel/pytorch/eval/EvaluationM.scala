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

import scala.language.implicitConversions

// evaluation for multi-labels
private[pytorch]
abstract class EvaluationM extends Serializable {

  def calculate(pairs: RDD[(Double, Double)]): String
}

private[pytorch]
object EvaluationM {

  def eval(metrics: Array[String], pairs: RDD[(Double, Double)], numLabels: Int = 1): Map[String, String] = {
    metrics.map(name => (name.toLowerCase(), EvaluationM.apply(name, numLabels).calculate(pairs))).toMap
  }

  def apply(name: String, numLabels: Int = 1): EvaluationM = {
    name.toLowerCase match {
      case "multi_auc" => new MultiLabelAUC(numLabels)
      case "multi_auc_collect" => new MultiLabelAUCCollect(numLabels)
    }
  }

  implicit def pairNumericRDDToPairDoubleRDD[T](rdd: RDD[(T, T)])(implicit num: Numeric[T])
  : RDD[(Double, Double)] = {
    rdd.map(x => (num.toDouble(x._1), num.toDouble(x._2)))
  }
}