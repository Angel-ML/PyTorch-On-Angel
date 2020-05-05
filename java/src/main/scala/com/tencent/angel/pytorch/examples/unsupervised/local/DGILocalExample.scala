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
package com.tencent.angel.pytorch.examples.unsupervised.local

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.pytorch.graph.gcn.DGI
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object DGILocalExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val featureInput = params.getOrElse("featurePath", "")
    val embeddingPath = params.getOrElse("embeddingPath", "")
    val batchSize = params.getOrElse("batchSize", "100").toInt
    val torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val featureDim = params.getOrElse("featureDim", "-1").toInt
    val optimizer = params.getOrElse("optimizer", "adam")
    val psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    val numPartitions = params.getOrElse("numPartitions", "5").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val second = params.getOrElse("second", "false").toBoolean

    start()

    val dgi = new DGI()
    dgi.setTorchModelPath(torchModelPath)
    dgi.setFeatureDim(featureDim)
    dgi.setOptimizer(optimizer)
    dgi.setBatchSize(batchSize)
    dgi.setStepSize(stepSize)
    dgi.setPSPartitionNum(psNumPartition)
    dgi.setPartitionNum(numPartitions)
    dgi.setUseBalancePartition(useBalancePartition)
    dgi.setNumEpoch(numEpoch)
    dgi.setStorageLevel(storageLevel)
    dgi.setUseSecondOrder(second)

    val edges = GraphIO.load(edgeInput, isWeighted = false)
    val features = IOFunctions.loadFeature(featureInput, sep = "\t")

    val (model, graph) = dgi.initialize(edges, features)
    dgi.fit(model, graph)

    if (embeddingPath.length > 0) {
      val embedding = dgi.genEmbedding(model, graph)
      GraphIO.save(embedding, embeddingPath, seq = " ")
    }

    stop()
  }


  def start(mode: String = "local"): Unit = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("gcn")
    conf.set(AngelConf.ANGEL_PSAGENT_UPDATE_SPLIT_ADAPTION_ENABLE, "false")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("cp")
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}
