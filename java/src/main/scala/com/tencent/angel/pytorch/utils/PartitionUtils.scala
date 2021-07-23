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
package com.tencent.angel.pytorch.utils

import com.tencent.angel.spark.ml.util.SparkUtils
import org.apache.spark.SparkConf

object PartitionUtils {

  def getDataPartitionNum(numPartitions: Int, conf: SparkConf, factor: Int = 3): Int = {
    val cores = SparkUtils.getNumCores(conf)
    if (numPartitions > cores * factor) numPartitions else cores * factor
  }

  def getPsPartitionNum(psNumPartitions: Int, conf: SparkConf, factor: Int = 2): Int = {
    val numPs = conf.get("spark.ps.instances").toInt
    val numPsCores = conf.get("spark.ps.cores").toInt
    if (psNumPartitions > numPs * numPsCores * factor) psNumPartitions else numPs * numPsCores * factor
  }
}