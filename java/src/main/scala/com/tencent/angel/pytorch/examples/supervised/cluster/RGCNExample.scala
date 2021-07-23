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
package com.tencent.angel.pytorch.examples.supervised.cluster

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.pytorch.graph.gcn.RGCN
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object RGCNExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    val edgeInput = params.getOrElse("edgePath", "")
    val featureInput = params.getOrElse("featurePath", "")
    val labelPath = params.getOrElse("labelPath", "")
    val predictOutputPath = params.getOrElse("predictOutputPath", "")
    val embeddingPath = params.getOrElse("embeddingPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val fieldNum = params.getOrElse("fieldNum", "-1").toInt
    val featEmbedDim = params.getOrElse("featEmbedDim", "-1").toInt
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
    val batchSize = params.getOrElse("batchSize", "100").toInt
    var torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val featureDim = params.getOrElse("featureDim", "-1").toInt
    val optimizer = params.getOrElse("optimizer", "adam")
    var psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    var numPartitions = params.getOrElse("numPartitions", "10").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "2").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "3").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val testRatio = params.getOrElse("testRatio", "0.5").toFloat
    val format = params.getOrElse("format", "sparse")
    val numSamples = params.getOrElse("samples", "5").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val numBatchInit = params.getOrElse("numBatchInit", "5").toInt
    val actionType = params.getOrElse("actionType", "train")
    val checkpointInterval = params.getOrElse("checkpointInterval", "0").toInt
    var evals = params.getOrElse("evals", "acc")
    val useSecondOrder = params.getOrElse("second", "true").toBoolean
    var useSharedSamples = params.getOrElse("useSharedSamples", "false").toBoolean
    if (batchSize < 128) useSharedSamples = false
    val numLabels = params.getOrElse("numLabels", "1").toInt // a multi-label classification task if numLabels > 1
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    val sep = params.getOrElse("sep", "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }

    if (numLabels > 1) evals = "multi_auc"

    val conf = start(mode)

    numPartitions = PartitionUtils.getDataPartitionNum(numPartitions, conf, numPartitionsFactor)
    psNumPartition = PartitionUtils.getPsPartitionNum(psNumPartition, conf, psNumPartitionFactor)
    println(s"numPartition: $numPartitions, numPsPartition: $psNumPartition")

    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)

    val rgcn = new RGCN()
    rgcn.setTorchModelPath(torchModelPath)
    rgcn.setFeatureDim(featureDim)
    rgcn.setOptimizer(optimizer)
    rgcn.setUseBalancePartition(false)
    rgcn.setBatchSize(batchSize)
    rgcn.setStepSize(stepSize)
    rgcn.setPSPartitionNum(psNumPartition)
    rgcn.setPartitionNum(numPartitions)
    rgcn.setUseBalancePartition(useBalancePartition)
    rgcn.setNumEpoch(numEpoch)
    rgcn.setStorageLevel(storageLevel)
    rgcn.setTestRatio(testRatio)
    rgcn.setDataFormat(format)
    rgcn.setNumSamples(numSamples)
    rgcn.setNumBatchInit(numBatchInit)
    rgcn.setCheckpointInterval(checkpointInterval)
    rgcn.setUseSecondOrder(useSecondOrder)
    rgcn.setUseSharedSamples(useSharedSamples)
    rgcn.setNumLabels(numLabels)
    rgcn.setEvaluations(evals)
    rgcn.setBatchSizeMultiple(batchSizeMultiple)
    rgcn.setFeatEmbedPath(featureEmbedInputPath)
    rgcn.setFeatEmbedDim(featEmbedDim)
    rgcn.setFieldNum(fieldNum)
    rgcn.setFieldMultiHot(fieldMultiHot)

    val edges = IOFunctions.loadEdge(edgeInput, isTyped = true, sep = sep)
    val features = IOFunctions.loadFeature(featureInput, sep = "\t")
    val labels = if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath)

    DataStatistics.setAppName(rgcn.getClass.getCanonicalName)
    DataStatistics.calcRDDSize(edges, "angel.input.edges.data")
    DataStatistics.calcRDDSize(features, "angel.input.features.data")
    DataStatistics.calcRDDSize(labels, "angel.input.labels.data")

    val (model, graph) = rgcn.initialize(edges, features, Option(labels))
    rgcn.showSummary(model, graph)

    if (actionType == "train")
      rgcn.fit(model, graph)

    if (predictOutputPath.length > 0) {
      val predict = rgcn.genLabels(model, graph)
      GraphIO.save(predict, predictOutputPath, seq = " ")
    }

    if (embeddingPath.length > 0) {
      val embedding = rgcn.genEmbedding(model, graph)
      GraphIO.save(embedding, embeddingPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      rgcn.save(model, outputModelPath)
      if (fieldNum > 0) {
        rgcn.saveFeatEmbed(model, outputModelPath)
      }
    }

    stop()
  }

  def start(mode: String = "local"): SparkConf = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("r-gcn")
    conf.set(AngelConf.ANGEL_PSAGENT_UPDATE_SPLIT_ADAPTION_ENABLE, "false")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("cp")
    conf
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}
