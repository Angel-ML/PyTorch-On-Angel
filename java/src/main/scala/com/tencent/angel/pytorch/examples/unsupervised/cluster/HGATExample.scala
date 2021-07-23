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
package com.tencent.angel.pytorch.examples.unsupervised.cluster

import com.tencent.angel.pytorch.graph.gcn.HGAT
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

object HGATExample {
  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val userFeatureInput = params.getOrElse("userFeaturePath", "")
    val itemFeatureInput = params.getOrElse("itemFeaturePath", "")
    val userEmbeddingPath = params.getOrElse("userEmbeddingPath", "")
    val itemEmbeddingPath = params.getOrElse("itemEmbeddingPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val userFieldNum = params.getOrElse("userFieldNum", "-1").toInt
    val itemFieldNum = params.getOrElse("itemFieldNum", "-1").toInt
    val userFeatEmbedDim = params.getOrElse("userFeatEmbedDim", "-1").toInt
    val itemFeatEmbedDim = params.getOrElse("itemFeatEmbedDim", "-1").toInt
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
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
    val trainRatio = params.getOrElse("trainRatio", "0.5").toFloat
    val userNumSamples = params.getOrElse("userNumSamples", "5").toInt
    val itemNumSamples = params.getOrElse("itemNumSamples", "5").toInt
    val sep = params.getOrElse("sep", "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }
    val featureSep = params.getOrElse("featureSep", "tab") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }


    val conf = new SparkConf()
    conf.set("spark.executorEnv.OMP_NUM_THREADS", "2")
    conf.set("spark.executorEnv.MKL_NUM_THREADS", "2")

    val sc = new SparkContext(conf)

    numPartitions = PartitionUtils.getDataPartitionNum(numPartitions, conf, numPartitionsFactor)
    psNumPartition = PartitionUtils.getPsPartitionNum(psNumPartition, conf, psNumPartitionFactor)
    println(s"numPartition: $numPartitions, numPsPartition: $psNumPartition")

    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)

    val hgat = new HGAT()
    hgat.setTorchModelPath(torchModelPath)
    hgat.setUserFeatureDim(userFeatureDim)
    hgat.setItemFeatureDim(itemFeatureDim)
    hgat.setOptimizer(optimizer)
    hgat.setBatchSize(batchSize)
    hgat.setStepSize(stepSize)
    hgat.setPSPartitionNum(psNumPartition)
    hgat.setPartitionNum(numPartitions)
    hgat.setUseBalancePartition(useBalancePartition)
    hgat.setNumEpoch(numEpoch)
    hgat.setStorageLevel(storageLevel)
    hgat.setUseSecondOrder(second)
    hgat.setDataFormat(format)
    hgat.setSaveCheckpoint(saveCheckpoint)
    hgat.setPeriods(periods)
    hgat.setDecay(decay)
    hgat.setBatchSizeMultiple(batchSizeMultiple)
    hgat.setFeatEmbedPath(featureEmbedInputPath)
    hgat.setUserFeatEmbedDim(userFeatEmbedDim)
    hgat.setUserFieldNum(userFieldNum)
    hgat.setItemFeatEmbedDim(itemFeatEmbedDim)
    hgat.setItemFieldNum(itemFieldNum)
    hgat.setTestRatio(trainRatio)
    hgat.setFieldMultiHot(fieldMultiHot)
    hgat.setUserNumSamples(userNumSamples)
    hgat.setItemNumSamples(itemNumSamples)

    val edges = GraphIO.load(edgeInput, isWeighted = false, sep = sep)
    val userFeatures = IOFunctions.loadFeature(userFeatureInput, sep = featureSep)
    val itemFeatures = if (itemFeatureDim > 0) IOFunctions.loadFeature(itemFeatureInput, sep = featureSep) else null

    val (model, userGraph, itemGraph) = hgat.initialize(edges, userFeatures, itemFeatures)

    hgat.showSummary(model, userGraph, itemGraph)
    if (actionType == "train")
      hgat.fit(model, userGraph, itemGraph, outputModelPath)

    if (userEmbeddingPath.length > 0) {
      val userEmbedding = hgat.genEmbedding(model, userGraph, 0)
      if (makeDenseOutput) {
        val result = userEmbedding.rdd.map(row => row.getLong(0) + " " + row.getString(1).split(",").mkString(" "))
        result.saveAsTextFile(userEmbeddingPath)
      } else {
        GraphIO.save(userEmbedding, userEmbeddingPath, seq = " ")
      }

    }
    if (itemEmbeddingPath.length > 0) {
      val itemEmbedding = hgat.genEmbedding(model, itemGraph, 1)
      if (makeDenseOutput) {
        val result = itemEmbedding.rdd.map(row => row.getLong(0) + " " + row.getString(1).split(",").mkString(" "))
        result.saveAsTextFile(itemEmbeddingPath)
      } else {
        GraphIO.save(itemEmbedding, itemEmbeddingPath, seq = " ")
      }
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      hgat.save(model, outputModelPath)
      if (userFieldNum > 0) {
        hgat.saveFeatEmbed(model, outputModelPath)
      }
    }

    PSContext.stop()
    sc.stop()
  }

}
