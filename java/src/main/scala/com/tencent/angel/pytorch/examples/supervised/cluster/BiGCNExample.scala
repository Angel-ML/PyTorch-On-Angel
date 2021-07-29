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

import com.tencent.angel.pytorch.graph.gcn.BiGCN
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object BiGCNExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    val edgeInput = params.getOrElse("edgePath", "")
    val userFeatureInput = params.getOrElse("userFeaturePath", "")
    val itemFeatureInput = params.getOrElse("itemFeaturePath", "")
    val userEmbeddingPath = params.getOrElse("userEmbeddingPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val userFieldNum = params.getOrElse("userFieldNum", "-1").toInt
    val itemFieldNum = params.getOrElse("itemFieldNum", "-1").toInt
    val userFeatEmbedDim = params.getOrElse("userFeatEmbedDim", "-1").toInt
    val itemFeatEmbedDim = params.getOrElse("itemFeatEmbedDim", "-1").toInt
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
    val labelPath = params.getOrElse("labelPath", "")
    val testLabelPath = params.getOrElse("testLabelPath", "")
    val batchSize = params.getOrElse("batchSize", "100").toInt
    var torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val testRatio = params.getOrElse("testRatio", "0.5").toFloat
    val userFeatureDim = params.getOrElse("userFeatureDim", "-1").toInt
    val itemFeatureDim = params.getOrElse("itemFeatureDim", "-1").toInt
    val optimizer = params.getOrElse("optimizer", "adam")
    var psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    var numPartitions = params.getOrElse("numPartitions", "5").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "2").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "3").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val format = params.getOrElse("format", "sparse")
    val userNumSamples = params.getOrElse("userNumSamples", "5").toInt
    val itemNumSamples = params.getOrElse("itemNumSamples", "5").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val numBatchInit = params.getOrElse("numBatchInit", "5").toInt
    val actionType = params.getOrElse("actionType", "train")
    val useSecondOrder = params.getOrElse("second", "false").toBoolean
    val periods = params.getOrElse("periods", "1000").toInt
    val checkpointInterval = params.getOrElse("checkpointInterval", "0").toInt
    val decay = params.getOrElse("decay", "0.000").toFloat
    var evals = params.getOrElse("evals", "acc")
    val validatePeriods = params.getOrElse("validatePeriods", "1").toInt
    val saveCheckpoint = params.getOrElse("saveCheckpoint", "false").toBoolean
    val hasNodeType = params.getOrElse("hasNodeType", "false").toBoolean
    val hasEdgeType = params.getOrElse("hasEdgeType", "false").toBoolean
    var useSharedSamples = params.getOrElse("useSharedSamples", "true").toBoolean
    if (batchSize < 128 || userFieldNum > 0) useSharedSamples = false
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    val numLabels = params.getOrElse("numLabels", "1").toInt // a multi-label classification task if numLabels > 1
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

    val bisage = new BiGCN()
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
    bisage.setDataFormat(format)
    bisage.setTestRatio(testRatio)
    bisage.setUserNumSamples(userNumSamples)
    bisage.setItemNumSamples(itemNumSamples)
    bisage.setNumBatchInit(numBatchInit)
    bisage.setPeriods(periods)
    bisage.setCheckpointInterval(checkpointInterval)
    bisage.setDecay(decay)
    bisage.setEvaluations(evals)
    bisage.setValidatePeriods(validatePeriods)
    bisage.setUseSecondOrder(useSecondOrder)
    bisage.setSaveCheckpoint(saveCheckpoint)
    bisage.setHasNodeType(hasNodeType)
    bisage.setHasEdgeType(hasEdgeType)
    bisage.setUseSharedSamples(useSharedSamples)
    bisage.setNumLabels(numLabels)
    bisage.setBatchSizeMultiple(batchSizeMultiple)
    bisage.setFeatEmbedPath(featureEmbedInputPath)
    bisage.setUserFeatEmbedDim(userFeatEmbedDim)
    bisage.setUserFieldNum(userFieldNum)
    bisage.setItemFeatEmbedDim(itemFeatEmbedDim)
    bisage.setItemFieldNum(itemFieldNum)
    bisage.setFieldMultiHot(fieldMultiHot)

    val edges = GraphIO.loadWithType(edgeInput, isEdgeType = hasEdgeType, isSrcNodeType = false, isDstNodeType = hasNodeType, sep = sep)
    val userFeatures = IOFunctions.loadFeature(userFeatureInput, sep = "\t")
    val itemFeatures = if (itemFeatureDim > 0) IOFunctions.loadFeature(itemFeatureInput, sep = "\t") else null
    val labels = if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath)
    val testLabels = if (testLabelPath.length > 0)
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(testLabelPath, sep = "p") else IOFunctions.loadLabel(testLabelPath))
    else None

    val (model, userGraph, itemGraph) = bisage.initialize(edges, userFeatures, itemFeatures, Option(labels), testLabels)
    bisage.showSummary(model, userGraph, itemGraph)
    if (actionType == "train")
      bisage.fit(model, userGraph, itemGraph, outputModelPath)

    if (userEmbeddingPath.length > 0) {
      val userEmbedding = bisage.genLabelsEmbedding(model, userGraph)
      GraphIO.save(userEmbedding, userEmbeddingPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      bisage.save(model, outputModelPath)
      if (userFieldNum > 0) {
        bisage.saveFeatEmbed(model, outputModelPath)
      }
    }

    stop()
  }

  def start(mode: String = "local"): SparkConf = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("BiGCN")
    val sc = new SparkContext(conf)
    if (sc.isLocal)
      sc.setLogLevel("ERROR")
    conf
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}