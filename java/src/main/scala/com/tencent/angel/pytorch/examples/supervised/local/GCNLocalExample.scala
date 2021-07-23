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
package com.tencent.angel.pytorch.examples.supervised.local

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.pytorch.graph.gcn.GCN
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.PartitionUtils
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.existentials

object GCNLocalExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    val edgeInput = params.getOrElse("edgePath", "")
    val featureInput = params.getOrElse("featurePath", "")
    val labelPath = params.getOrElse("labelPath", "")
    val testLabelPath = params.getOrElse("testLabelPath", "")
    val predictOutputPath = params.getOrElse("predictOutputPath", "")
    val embeddingPath = params.getOrElse("embeddingPath", "")
    val outputModelPath = params.getOrElse("outputModelPath", "")
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val fieldNum = params.getOrElse("fieldNum", "-1").toInt
    val featEmbedDim = params.getOrElse("featEmbedDim", "-1").toInt
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
    val periods = params.getOrElse("periods", "1000").toInt
    val checkpointInterval = params.getOrElse("checkpointInterval", "0").toInt
    val decay = params.getOrElse("decay", "0.000").toFloat
    var evals = params.getOrElse("evals", "acc")
    val validatePeriods = params.getOrElse("validatePeriods", "5").toInt
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
    println(s"numPartition: $numPartitions, numPsPartition: $psNumPartition")

    val gcn = new GCN()
    gcn.setTorchModelPath(torchModelPath)
    gcn.setFeatureDim(featureDim)
    gcn.setOptimizer(optimizer)
    gcn.setBatchSize(batchSize)
    gcn.setStepSize(stepSize)
    gcn.setPSPartitionNum(psNumPartition)
    gcn.setPartitionNum(numPartitions)
    gcn.setUseBalancePartition(useBalancePartition)
    gcn.setNumEpoch(numEpoch)
    gcn.setStorageLevel(storageLevel)
    gcn.setTestRatio(testRatio)
    gcn.setDataFormat(format)
    gcn.setNumSamples(numSamples)
    gcn.setNumBatchInit(numBatchInit)
    gcn.setPeriods(periods)
    gcn.setCheckpointInterval(checkpointInterval)
    gcn.setDecay(decay)
    gcn.setEvaluations(evals)
    gcn.setValidatePeriods(validatePeriods)
    gcn.setUseSharedSamples(useSharedSamples)
    gcn.setUseSecondOrder(useSecondOrder)
    gcn.setNumLabels(numLabels)
    gcn.setBatchSizeMultiple(batchSizeMultiple)
    gcn.setFeatEmbedPath(featureEmbedInputPath)
    gcn.setFeatEmbedDim(featEmbedDim)
    gcn.setFieldNum(fieldNum)

    val edges = GraphIO.load(edgeInput, isWeighted = false)
    val features = IOFunctions.loadFeature(featureInput, sep = "\t")
    val labels = IOFunctions.loadLabel(labelPath)

    val (model, graph) = gcn.initialize(edges, features, Option(labels))

    if (actionType == "train")
      gcn.fit(model, graph)

    if (predictOutputPath.length > 0) {
      val embedPred = gcn.genLabelsEmbedding(model, graph)
      GraphIO.save(embedPred, predictOutputPath, seq = " ")
    }

    if (actionType == "train" && outputModelPath.length > 0) {
      gcn.save(model, outputModelPath)
      if (fieldNum > 0) {
        gcn.saveFeatEmbed(model, outputModelPath)
      }
    }

    stop()
  }


  def start(mode: String = "local"): SparkConf = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("gcn")
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
