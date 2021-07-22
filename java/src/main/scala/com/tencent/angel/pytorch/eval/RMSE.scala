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

import scala.math.{pow, sqrt}

class RMSE extends Evaluation {
  override def calculate(pairs: RDD[(Double, Double)]): Double = {
    val (se, total) = pairs.map(f => (pow(f._2 - f._1, 2), 1.0)).reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
    sqrt(se / total)
  }
}
