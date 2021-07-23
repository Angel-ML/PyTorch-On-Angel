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
package com.tencent.angel.pytorch.examples

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.graph.utils.GraphIO
import com.tencent.angel.pytorch.graph.egonetwork.EgoNetwork
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import org.apache.spark.{SparkConf, SparkContext}

object EgoNetworkExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgeInput", "")
    val labelPath = params.getOrElse("labelPath", "")

    val edgeOutput = params.getOrElse("edgeOutput", "")

    val numPartitions = params.getOrElse("numPartition", "2").toInt
    val psNumPartition = params.getOrElse("psNumPartition", "1").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val isUndirected = params.getOrElse("isUndirected", "false").toBoolean
    val sep = params.getOrElse("sep",  "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }

    val mode = params.getOrElse("mode", "local")

    start(mode)

    val sub = new EgoNetwork()
    sub.setLabelPath(labelPath)
    sub.setUseBalancePartition(useBalancePartition)
    sub.setPartitionNum(numPartitions)
    sub.setPSPartitionNum(psNumPartition)
    sub.setStorageLevel(storageLevel)
    sub.setStorageLevel(storageLevel)
    sub.setUnDirected(isUndirected)
    sub.setNumBatchInit(1)

    val edges = GraphIO.load(edgeInput, isWeighted = false, sep = sep)
    val edgeSamples = sub.transform(edges)
    GraphIO.save(edgeSamples, edgeOutput, seq = " ")

    stop()
  }

  def start(mode: String): Unit = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("EgoNetwork")
    conf.set(AngelConf.ANGEL_PSAGENT_UPDATE_SPLIT_ADAPTION_ENABLE, "false")
    new SparkContext(conf)
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}

