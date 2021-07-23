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

import com.tencent.angel.pytorch.graph.gcn.IGMC
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object IGMCExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val userFeatureInput = params.getOrElse("userFeaturePath", "")
    val itemFeatureInput = params.getOrElse("itemFeaturePath", "")
    val predictOutputPath = params.getOrElse("predictOutputPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val batchSize = params.getOrElse("batchSize", "100").toInt
    var torchModelPath = params.getOrElse("upload_torchModelPath", "model.pt")
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
    val numSamples = params.getOrElse("samples", "5").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val numBatchInit = params.getOrElse("numBatchInit", "5").toInt
    val actionType = params.getOrElse("actionType", "train")
    val useSecondOrder = params.getOrElse("second", "true").toBoolean
    val periods = params.getOrElse("periods", "1000").toInt
    val checkpointInterval = params.getOrElse("checkpointInterval", "0").toInt
    val decay = params.getOrElse("decay", "0.000").toFloat
    val evals = params.getOrElse("evals", "acc")
    val validatePeriods = params.getOrElse("validatePeriods", "1").toInt
    val saveCheckpoint = params.getOrElse("saveCheckpoint", "false").toBoolean
    val hasEdgeType = params.getOrElse("hasEdgeType", "false").toBoolean
    val taskType = params.getOrElse("taskType", "classification")
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    var useSharedSamples = params.getOrElse("useSharedSamples", "false").toBoolean
    if (batchSize < 128) useSharedSamples = false
    val sep = params.getOrElse("sep", "space") match {
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

    val igmc = new IGMC()
    igmc.setTorchModelPath(torchModelPath)
    igmc.setUserFeatureDim(userFeatureDim)
    igmc.setItemFeatureDim(itemFeatureDim)
    igmc.setOptimizer(optimizer)
    igmc.setBatchSize(batchSize)
    igmc.setStepSize(stepSize)
    igmc.setPSPartitionNum(psNumPartition)
    igmc.setPartitionNum(numPartitions)
    igmc.setUseBalancePartition(useBalancePartition)
    igmc.setNumEpoch(numEpoch)
    igmc.setStorageLevel(storageLevel)
    igmc.setDataFormat(format)
    igmc.setTestRatio(testRatio)
    igmc.setNumSamples(numSamples)
    igmc.setNumBatchInit(numBatchInit)
    igmc.setPeriods(periods)
    igmc.setCheckpointInterval(checkpointInterval)
    igmc.setDecay(decay)
    igmc.setEvaluations(evals)
    igmc.setValidatePeriods(validatePeriods)
    igmc.setUseSecondOrder(useSecondOrder)
    igmc.setSaveCheckpoint(saveCheckpoint)
    igmc.setHasEdgeType(hasEdgeType)
    igmc.setTaskType(taskType)
    igmc.setUseSharedSamples(useSharedSamples)
    igmc.setBatchSizeMultiple(batchSizeMultiple)


    val edges = GraphIO.load(edgeInput, isWeighted = hasEdgeType, sep = sep)
    val userFeatures = if (userFeatureDim > 0) IOFunctions.loadFeature(userFeatureInput, sep = "\t") else null
    val itemFeatures = if (itemFeatureDim > 0) IOFunctions.loadFeature(itemFeatureInput, sep = "\t") else null

    val (model, userGraph, itemGraph) = igmc.initialize(edges, userFeatures, itemFeatures, None, None)
    igmc.showSummary(model, userGraph, itemGraph)
    if (actionType == "train")
      igmc.fit(model, userGraph, itemGraph, outputModelPath)

    if (predictOutputPath.length > 0) {
      val predicts = igmc.genLabels(model, userGraph)
      GraphIO.save(predicts, predictOutputPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0)
      igmc.save(model, outputModelPath)

    PSContext.stop()
    sc.stop()
  }

  def start(mode: String = "local"): SparkConf = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("IGMC")
    val sc = new SparkContext(conf)
    if (sc.isLocal)
      sc.setLogLevel("ERROR")
    conf
  }
}