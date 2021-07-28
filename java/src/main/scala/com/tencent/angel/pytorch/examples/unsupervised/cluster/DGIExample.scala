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

import com.tencent.angel.pytorch.graph.gcn.DGI
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object DGIExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val featureInput = params.getOrElse("featurePath", "")
    val embeddingPath = params.getOrElse("embeddingPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val fieldNum = params.getOrElse("fieldNum", "-1").toInt
    val featEmbedDim = params.getOrElse("featEmbedDim", "-1").toInt
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
    val batchSize = params.getOrElse("batchSize", "100").toInt
    var torchModelPath = params.getOrElse("upload_torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val decay = params.getOrElse("decay", "0.0").toDouble
    val featureDim = params.getOrElse("featureDim", "-1").toInt
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
    val numSamples = params.getOrElse("samples", "5").toInt
    val numBatchInit = params.getOrElse("numBatchInit", "5").toInt
    val actionType = params.getOrElse("actionType", "train")
    val saveCheckpoint = params.getOrElse("saveCheckpoint", "false").toBoolean
    val periods = params.getOrElse("periods", "1000").toInt
    val isWeighted = params.getOrElse("isWeighted", "false").toBoolean
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    //random sample trainRatio samples to train in each epoch
    val trainRatio = params.getOrElse("trainRatio", "0.5").toFloat
    val sep = params.getOrElse("sep", "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }

    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    numPartitions = PartitionUtils.getDataPartitionNum(numPartitions, conf, numPartitionsFactor)
    psNumPartition = PartitionUtils.getPsPartitionNum(psNumPartition, conf, psNumPartitionFactor)
    println(s"numPartition: $numPartitions, numPsPartition: $psNumPartition")

    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)

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
    dgi.setDataFormat(format)
    dgi.setNumBatchInit(numBatchInit)
    dgi.setNumSamples(numSamples)
    dgi.setSaveCheckpoint(saveCheckpoint)
    dgi.setPeriods(periods)
    dgi.setHasWeighted(isWeighted)
    dgi.setDecay(decay)
    dgi.setBatchSizeMultiple(batchSizeMultiple)
    dgi.setFeatEmbedPath(featureEmbedInputPath)
    dgi.setFeatEmbedDim(featEmbedDim)
    dgi.setFieldNum(fieldNum)
    dgi.setTestRatio(trainRatio)
    dgi.setFieldMultiHot(fieldMultiHot)

    val edges = GraphIO.load(edgeInput, isWeighted = isWeighted, sep = sep)
    val features = IOFunctions.loadFeature(featureInput, sep = "\t")

    val (model, graph) = dgi.initialize(edges, features)
    dgi.showSummary(model, graph)
    if (actionType == "train")
      dgi.fit(model, graph)

    if (embeddingPath.length > 0) {
      val embedding = dgi.genEmbedding(model, graph)
      GraphIO.save(embedding, embeddingPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      dgi.save(model, outputModelPath)
      if (fieldNum > 0) {
        dgi.saveFeatEmbed(model, outputModelPath)
      }
    }

    PSContext.stop()
    sc.stop()
  }

}