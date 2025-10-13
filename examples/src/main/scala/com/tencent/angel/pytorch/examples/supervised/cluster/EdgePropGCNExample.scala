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

import com.tencent.angel.pytorch.graph.gcn.EdgePropGCN
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object EdgePropGCNExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    val edgeInput = params.getOrElse("edgePath", "")
    val featureInput = params.getOrElse("featurePath", "")
    val labelPath = params.getOrElse("labelPath", "")
    val testLabelPath = params.getOrElse("testLabelPath", "")
    val predictOutputPath = params.getOrElse("predictOutputPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val fieldNum = params.getOrElse("fieldNum", "-1").toInt
    val featEmbedDim = params.getOrElse("featEmbedDim", "-1").toInt
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
    val batchSize = params.getOrElse("batchSize", "100").toInt
    var torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val featureDim = params.getOrElse("featureDim", "-1").toInt
    val edgeFeatureDim = params.getOrElse("edgeFeatureDim", "-1").toInt
    val optimizer = params.getOrElse("optimizer", "adam")
    var psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    var numPartitions = params.getOrElse("numPartitions", "1").toInt
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
    val periods = params.getOrElse("periods", "1000").toInt
    val decay = params.getOrElse("decay", "0.000").toFloat
    var evals = params.getOrElse("evals", "acc")
    val useSecondOrder = params.getOrElse("second", "true").toBoolean
    var useSharedSamples = params.getOrElse("useSharedSamples", "false").toBoolean
    if (batchSize < 128) useSharedSamples = false
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    val numLabels = params.getOrElse("numLabels", "1").toInt // a multi-label classification task if numLabels > 1
    if (numLabels > 1) evals = "multi_auc"


    val conf = start(mode)

    numPartitions = PartitionUtils.getDataPartitionNum(numPartitions, conf, numPartitionsFactor)
    psNumPartition = PartitionUtils.getPsPartitionNum(psNumPartition, conf, psNumPartitionFactor)
    println(s"numPartition: $numPartitions, numPsPartition: $psNumPartition")

    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)

    val edgeProp = new EdgePropGCN()
    edgeProp.setTorchModelPath(torchModelPath)
    edgeProp.setFeatureDim(featureDim)
    edgeProp.setEdgeFeatureDim(edgeFeatureDim)
    edgeProp.setOptimizer(optimizer)
    edgeProp.setUseBalancePartition(false)
    edgeProp.setBatchSize(batchSize)
    edgeProp.setStepSize(stepSize)
    edgeProp.setPSPartitionNum(psNumPartition)
    edgeProp.setPartitionNum(numPartitions)
    edgeProp.setUseBalancePartition(useBalancePartition)
    edgeProp.setNumEpoch(numEpoch)
    edgeProp.setStorageLevel(storageLevel)
    edgeProp.setTestRatio(testRatio)
    edgeProp.setDataFormat(format)
    edgeProp.setNumSamples(numSamples)
    edgeProp.setNumBatchInit(numBatchInit)
    edgeProp.setPeriods(periods)
    edgeProp.setDecay(decay)
    edgeProp.setEvaluations(evals)
    edgeProp.setUseSecondOrder(useSecondOrder)
    edgeProp.setUseSharedSamples(useSharedSamples)
    edgeProp.setNumLabels(numLabels)
    edgeProp.setBatchSizeMultiple(batchSizeMultiple)
    edgeProp.setFeatEmbedPath(featureEmbedInputPath)
    edgeProp.setFeatEmbedDim(featEmbedDim)
    edgeProp.setFieldNum(fieldNum)
    edgeProp.setFieldMultiHot(fieldMultiHot)

    val edgeFeatures = IOFunctions.loadEdgeFeature(edgeInput, sep = "\t")
    val features = IOFunctions.loadFeature(featureInput, sep = "\t")
    val labels = if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath)
    val testLabels = if (testLabelPath.length > 0)
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(testLabelPath, sep = "p") else IOFunctions.loadLabel(testLabelPath))
    else None

    val (model, graph) = edgeProp.initialize(edgeFeatures, features, Option(labels), testLabels)
    edgeProp.showSummary(model, graph)

    if (actionType == "train")
      edgeProp.fit(model, graph, outputModelPath)

    if (predictOutputPath.length > 0) {
      val embedPred = edgeProp.genLabelsEmbedding(model, graph)
      GraphIO.save(embedPred, predictOutputPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      edgeProp.save(model, outputModelPath)
      if (fieldNum > 0) {
        edgeProp.saveFeatEmbed(model, outputModelPath)
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