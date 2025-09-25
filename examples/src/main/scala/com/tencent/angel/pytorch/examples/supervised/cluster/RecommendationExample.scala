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
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.recommendation.{RecommendPSModel, Recommendation}
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

object RecommendationExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val mode = params.getOrElse("mode", "yarn-cluster")
    val trainPath = params.getOrElse("trainInput", "")
    val validatePath = params.getOrElse("validateInput", "")
    val predictOutputPath = params.getOrElse("predictOutputPath", "")
    var torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val batchSize = params.getOrElse("batchSize", "512").toInt
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val optimizer = params.getOrElse("optimizer", "adam")
    val testRatio = params.getOrElse("testRatio", "0.1").toFloat
    val actionType = params.getOrElse("actionType", "train")
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val decay = params.getOrElse("decay", "0.001").toDouble
    val async = params.getOrElse("async", "true").toBoolean
    val numDataPartitions = params.getOrElse("numDataPartitions", "100").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "2").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "3").toInt
    val angelModelOutputPath = params.getOrElse("angelModelOutputPath", "")
    val angelModelInputPath = params.getOrElse("angelModelInputPath", "")
    val torchOutputModelPath = params.getOrElse("torchOutputModelPath", "")
    val rowType = params.getOrElse("rowType", "T_FLOAT_DENSE")
    val evals = params.getOrElse("evals", "auc")
    val level = params.getOrElse("storageLevel", "memory_only").toUpperCase()

    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)

    val recommendation = new Recommendation(torchModelPath)
    recommendation.setNumEpoch(numEpoch)
    recommendation.setStepSize(stepSize)
    recommendation.setOptimizer(optimizer)
    recommendation.setTestRatio(testRatio)
    recommendation.setBatchSize(batchSize)
    recommendation.setDecay(decay)
    recommendation.setAsync(async)
    recommendation.setEvaluations(evals)
    recommendation.setStorageLevel(StorageLevel.fromString(level))

    val conf = start(mode)
    val numPartitions = PartitionUtils.getDataPartitionNum(numDataPartitions, conf, numPartitionsFactor)
    println(s"numDataPartitions=$numPartitions")

    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    println(torch)
    // training should be libsvm/libffm format
    val trainInput = IOFunctions.loadString(trainPath).repartition(numPartitions)
    val optim = recommendation.getOptimizer
    println(s"optimizer: $optim")

    if (actionType == "train") {
      val model = RecommendPSModel.apply(torch, optim.getNumSlots(), rowType, angelModelInputPath)
      if (validatePath.length > 0) {
        val validateInput = IOFunctions.loadString(validatePath).repartition(numPartitions)
        recommendation.fit(model, trainInput, validateInput, optim)
      } else
        recommendation.fit(model, trainInput, optim)
      // model save
      if (angelModelOutputPath.length > 0)
        model.savePSModel(angelModelOutputPath)
      //pytorch model save
      if (torchOutputModelPath.length > 0)
        model.saveTorchModel(torchOutputModelPath)
    } else if (actionType == "predict") {
      // model load
      assert(angelModelInputPath.length > 0 && predictOutputPath.length > 0)
      val model = RecommendPSModel.apply(torch, optim.getNumSlots(), rowType, angelModelInputPath)
      val results = recommendation.predict(model, trainInput)
      GraphIO.save(results, predictOutputPath)
    }
    stop()
  }

  def start(mode: String = "local"): SparkConf = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("RecommendationExample")
    conf.set(AngelConf.ANGEL_PS_ROUTER_TYPE, "range")
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