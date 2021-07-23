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
import com.tencent.angel.pytorch.graph.gcn.BiSAGE
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object BiGraphSageLocalExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val userFeatureInput = params.getOrElse("userFeaturePath", "")
    val itemFeatureInput = params.getOrElse("itemFeaturePath", "")
    val userEmbeddingPath = params.getOrElse("userEmbeddingPath", "")
    val itemEmbeddingPath = params.getOrElse("itemEmbeddingPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val fieldNum = params.getOrElse("fieldNum", "-1").toInt
    val batchSize = params.getOrElse("batchSize", "100").toInt
    var torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val decay = params.getOrElse("decay", "0.0").toDouble
    val userFeatureDim = params.getOrElse("userFeatureDim", "-1").toInt
    val itemFeatureDim = params.getOrElse("itemFeatureDim", "-1").toInt
    val optimizer = params.getOrElse("optimizer", "adam")
    var psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    var numPartitions = params.getOrElse("numPartitions", "5").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "2").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "3").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val second = params.getOrElse("second", "false").toBoolean
    val format = params.getOrElse("format", "sparse")
    val actionType = params.getOrElse("actionType", "train")
    val saveCheckpoint = params.getOrElse("saveCheckpoint", "false").toBoolean
    val periods = params.getOrElse("periods", "1000").toInt
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    val makeDenseOutput = params.getOrElse("makeDenseOutput", "false").toBoolean

    start()

    val bisage = new BiSAGE()
    bisage.setTorchModelPath(torchModelPath)
    bisage.setUserFeatureDim(userFeatureDim)
    bisage.setItemFeatureDim(itemFeatureDim)
    bisage.setOptimizer(optimizer)
    bisage.setBatchSize(batchSize)
    bisage.setStepSize(stepSize)
    bisage.setPSPartitionNum(psNumPartition)
    bisage.setPartitionNum(numPartitions)
    bisage.setUseBalancePartition(useBalancePartition)
    bisage.setNumEpoch(numEpoch)
    bisage.setStorageLevel(storageLevel)
    bisage.setUseSecondOrder(second)
    bisage.setDataFormat(format)
    bisage.setBatchSizeMultiple(batchSizeMultiple)
    bisage.setFeatEmbedPath(featureEmbedInputPath)

    val edges = GraphIO.load(edgeInput, isWeighted = true)
    val userFeatures = IOFunctions.loadFeature(userFeatureInput, sep = "\t")
    val itemFeatures =  if (itemFeatureDim > 0)  IOFunctions.loadFeature(itemFeatureInput, sep = "\t") else null

    val (model, userGraph, itemGraph) = bisage.initialize(edges, userFeatures, itemFeatures)
    bisage.showSummary(model, userGraph, itemGraph)
    bisage.fit(model, userGraph, itemGraph)

    if (userEmbeddingPath.length > 0) {
      val userEmbedding = bisage.genEmbedding(model, userGraph, 0)
      GraphIO.save(userEmbedding, userEmbeddingPath, seq = " ")
    }
    if (itemEmbeddingPath.length > 0) {
      val itemEmbedding = bisage.genEmbedding(model, itemGraph, 1)
      GraphIO.save(itemEmbedding, itemEmbeddingPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      bisage.save(model, outputModelPath)
      if (fieldNum > 0) {
        bisage.saveFeatEmbed(model, outputModelPath)
      }
    }


    stop()
  }

  def start(mode: String = "local"): Unit = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("gcn")
    conf.set(AngelConf.ANGEL_PSAGENT_UPDATE_SPLIT_ADAPTION_ENABLE, "false")
    conf.set("spark.hadoop." + AngelConf.ANGEL_PS_BACKUP_AUTO_ENABLE, "false")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("cp")
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}