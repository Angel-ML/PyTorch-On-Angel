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

import com.tencent.angel.pytorch.graph.gcn.hetAttention.HAN
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.PartitionUtils
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object HANExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    val edgeInput = params.getOrElse("edgePath", "")
    val userFeatureInput = params.getOrElse("userFeaturePath", "")
    val itemFeatureInput = params.getOrElse("itemFeaturePath", "")
    val userEmbeddingPath = params.getOrElse("userEmbeddingPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val fieldNum = params.getOrElse("fieldNum", "-1").toInt
    val featEmbedDim = params.getOrElse("featEmbedDim", "-1").toInt
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
    val labelPath = params.getOrElse("labelPath", "")
    val testLabelPath = params.getOrElse("testLabelPath", "")
    val batchSize = params.getOrElse("batchSize", "100").toInt
    val torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val testRatio = params.getOrElse("testRatio", "0.01").toFloat
    val userFeatureDim = params.getOrElse("userFeatureDim", "2").toInt
    val itemFeatureDim = params.getOrElse("itemFeatureDim", "-1").toInt
    val optimizer = params.getOrElse("optimizer", "adam")
    var psNumPartition = params.getOrElse("psNumPartition", "1").toInt
    var numPartitions = params.getOrElse("numPartitions", "1").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "1").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "1").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val format = params.getOrElse("format", "dense")
    val numSamples = params.getOrElse("samples", "5").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val numBatchInit = params.getOrElse("numBatchInit", "10").toInt
    val actionType = params.getOrElse("actionType", "train")
    val useSecondOrder = params.getOrElse("second", "false").toBoolean
    val periods = params.getOrElse("periods", "1000").toInt
    val checkpointInterval = params.getOrElse("checkpointInterval", "0").toInt
    val decay = params.getOrElse("decay", "0.000").toFloat
    var evals = params.getOrElse("evals", "acc")
    val validatePeriods = params.getOrElse("validatePeriods", "1").toInt
    val saveCheckpoint = params.getOrElse("saveCheckpoint", "false").toBoolean
    val hasNodeType = params.getOrElse("hasNodeType", "true").toBoolean
    val itemTypes = params.getOrElse("itemTypes", "266").toInt
    val numLabels = params.getOrElse("numLabels", "1").toInt // a multi-label classification task if numLabels > 1
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    // sep for reading edges
    val sep = params.getOrElse("sep",  "tab") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }

    if (numLabels > 1) evals = "multi_auc"

    val conf = start(mode)

    numPartitions = PartitionUtils.getDataPartitionNum(numPartitions, conf, numPartitionsFactor)
    psNumPartition = PartitionUtils.getPsPartitionNum(psNumPartition, conf, psNumPartitionFactor)
    println(s"numPartition: $numPartitions, numPsPartition: $psNumPartition")

    val han = new HAN()
    han.setTorchModelPath(torchModelPath)
    han.setUserFeatureDim(userFeatureDim)
    han.setOptimizer(optimizer)
    han.setBatchSize(batchSize)
    han.setStepSize(stepSize)
    han.setPSPartitionNum(psNumPartition)
    han.setPartitionNum(numPartitions)
    han.setUseBalancePartition(useBalancePartition)
    han.setNumEpoch(numEpoch)
    han.setStorageLevel(storageLevel)
    han.setDataFormat(format)
    han.setTestRatio(testRatio)
    han.setNumSamples(numSamples)
    han.setNumBatchInit(numBatchInit)
    han.setPeriods(periods)
    han.setCheckpointInterval(checkpointInterval)
    han.setDecay(decay)
    han.setEvaluations(evals)
    han.setValidatePeriods(validatePeriods)
    han.setUseSecondOrder(useSecondOrder)
    han.setSaveCheckpoint(saveCheckpoint)
    han.setHasNodeType(hasNodeType)
    han.setItemTypes(itemTypes)
    han.setFeatureDim(userFeatureDim)
    han.setNumLabels(numLabels)
    han.setBatchSizeMultiple(batchSizeMultiple)
    han.setFeatEmbedPath(featureEmbedInputPath)
    han.setFeatEmbedDim(featEmbedDim)
    han.setFieldNum(fieldNum)
    han.setFieldMultiHot(fieldMultiHot)

    val edges = GraphIO.load(edgeInput, isWeighted = hasNodeType, sep = sep)
    val userFeatures = IOFunctions.loadFeature(userFeatureInput, sep = "\t")
    val labels = if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath)
    val testLabels = if (testLabelPath.length > 0)
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(testLabelPath, sep = "p") else IOFunctions.loadLabel(testLabelPath))
    else None

    val (model, graph) = han.initialize(edges, userFeatures, Option(labels), testLabels)
    han.showSummary(model, graph)
    if (actionType == "train")
      han.fit(model, graph)

    if (userEmbeddingPath.length > 0) {
      val userEmbedding = han.genLabelsEmbedding(model, graph)
      GraphIO.save(userEmbedding, userEmbeddingPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      han.save(model, outputModelPath)
      if (fieldNum > 0) {
        han.saveFeatEmbed(model, outputModelPath)
      }
    }

    stop()
  }

  def start(mode: String = "local"): SparkConf = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("EdgeProp")
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