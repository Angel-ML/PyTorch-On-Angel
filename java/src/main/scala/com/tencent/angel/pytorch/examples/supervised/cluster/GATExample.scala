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

import com.tencent.angel.pytorch.graph.gcn.{GAT, GCN}
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.FileUtils
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.SparkContext

import scala.language.existentials

object GATExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
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
    val optimizer = params.getOrElse("optimizer", "adam")
    val psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    val numPartitions = params.getOrElse("numPartitions", "1").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val testRatio = params.getOrElse("testRatio", "0.5").toFloat
    val format = params.getOrElse("format", "sparse")
    val numSamples = params.getOrElse("samples", "5").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val numBatchInit = params.getOrElse("numBatchInit", "5").toInt
    val actionType = params.getOrElse("actionType", "train")
    val periods = params.getOrElse("periods", "1000").toInt
    val checkpointInterval = params.getOrElse("checkpointInterval", "0").toInt
    val decay = params.getOrElse("decay", "0.000").toFloat
    var evals = params.getOrElse("evals", "acc")
    val validatePeriods = params.getOrElse("validatePeriods", "1").toInt
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    val useSecondOrder = params.getOrElse("second", "false").toBoolean
    val numLabels = params.getOrElse("numLabels", "1").toInt // a multi-label classification task if numLabels > 1
    if (numLabels > 1) evals = "multi_auc"
    val sep = params.getOrElse("sep", "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }

    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)

    val gat = new GAT()
    gat.setTorchModelPath(torchModelPath)
    gat.setFeatureDim(featureDim)
    gat.setOptimizer(optimizer)
    gat.setUseBalancePartition(false)
    gat.setBatchSize(batchSize)
    gat.setStepSize(stepSize)
    gat.setPSPartitionNum(psNumPartition)
    gat.setPartitionNum(numPartitions)
    gat.setUseBalancePartition(useBalancePartition)
    gat.setNumEpoch(numEpoch)
    gat.setStorageLevel(storageLevel)
    gat.setTestRatio(testRatio)
    gat.setDataFormat(format)
    gat.setNumSamples(numSamples)
    gat.setNumBatchInit(numBatchInit)
    gat.setPeriods(periods)
    gat.setCheckpointInterval(checkpointInterval)
    gat.setDecay(decay)
    gat.setEvaluations(evals)
    gat.setValidatePeriods(validatePeriods)
    gat.setUseSecondOrder(useSecondOrder)
    gat.setNumLabels(numLabels)
    gat.setBatchSizeMultiple(batchSizeMultiple)
    gat.setFeatEmbedPath(featureEmbedInputPath)
    gat.setFeatEmbedDim(featEmbedDim)
    gat.setFieldNum(fieldNum)
    gat.setFieldMultiHot(fieldMultiHot)

    val edges = GraphIO.load(edgeInput, isWeighted = false, sep = sep)
    val features = IOFunctions.loadFeature(featureInput, sep = "\t")
    val labels = if (numLabels > 1) IOFunctions.loadMultiLabel(labelPath, sep = "p") else IOFunctions.loadLabel(labelPath)
    val testLabels = if (testLabelPath.length > 0)
      Option(if (numLabels > 1) IOFunctions.loadMultiLabel(testLabelPath, sep = "p") else IOFunctions.loadLabel(testLabelPath))
    else None

    val (model, graph) = gat.initialize(edges, features, Option(labels), testLabels)
    gat.showSummary(model, graph)

    if (actionType == "train")
      gat.fit(model, graph, outputModelPath)

    if (predictOutputPath.length > 0) {
      val embedPred = gat.genLabelsEmbedding(model, graph)
      GraphIO.save(embedPred, predictOutputPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      gat.save(model, outputModelPath)
      if (fieldNum > 0) {
        gat.saveFeatEmbed(model, outputModelPath)
      }
    }

    stop()
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}