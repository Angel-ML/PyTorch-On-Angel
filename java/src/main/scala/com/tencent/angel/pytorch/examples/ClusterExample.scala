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
package com.tencent.angel.pytorch.examples

import com.tencent.angel.pytorch.optim.AsyncAdam
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.core.metric.AUC
import com.tencent.angel.spark.ml.util.SparkUtils
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

object ClusterExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "")
    val batchSize = params.getOrElse("batchSize", "512").toInt
    val torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toFloat
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val validateRatio = params.getOrElse("validateRatio", "0.1").toFloat
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val modelPath = params.getOrElse("modelPath", "")
    val modulePath = params.getOrElse("modulePath", "")
    val action = params.getOrElse("action", "train")
    val predictOutput = params.getOrElse("predictOutput", "")
    val mode = params.getOrElse("mode", "")
    import com.tencent.angel.pytorch.model.ParTorchModel

    val conf = new SparkConf()
    if(mode == "local") {
      conf.setMaster("local[1]")
      conf.setAppName("local mode for debug.")
    }
    val sc = new SparkContext(conf)
    if(sc.isLocal) {
      sc.setLogLevel("ERROR")
    }

    PSContext.getOrCreate(sc)

    val numCores = SparkUtils.getNumCores(conf)
    val numDataPartitions = (numCores * 3.0).toInt
    if (action.equals("train")) {
      val data = sc.textFile(input).repartition(numDataPartitions)
      val splits = data.randomSplit(Array(1 - validateRatio, validateRatio))
      val (train, test) = (splits(0), splits(1))
      train.persist(StorageLevel.fromString(storageLevel))
      test.persist(StorageLevel.fromString(storageLevel))

      val optim = new AsyncAdam(stepSize)
      val model = new ParTorchModel(optim, torchModelPath)
      model.init()

      for (epoch <- 1 to numEpoch) {
        val epochStartTime = System.currentTimeMillis()
        val (lossSum, size) = train.mapPartitions {
          iterator =>
            iterator.sliding(batchSize, batchSize)
              .map(batch => (model.optimize(batch.toArray), batch.length))
        }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))

        val scores = test.mapPartitions {
          iterator =>
            iterator.sliding(batchSize, batchSize)
              .map(batch => model.predict(batch.toArray))
              .flatMap(f => f._1.zip(f._2))
              .map(f => (f._1.toDouble, f._2.toDouble))
        }

        val auc = new AUC().calculate(scores)
        val epochTime = System.currentTimeMillis() - epochStartTime
        println(s"epoch=$epoch loss=${lossSum / size} auc=$auc time=${epochTime.toFloat / 1000}s")
      }

      if (modelPath.length > 0)
        model.save(modelPath)
      if (modulePath.length > 0)
        model.saveModule(modulePath, SparkHadoopUtil.get.newConfiguration(sc.getConf))
    } else {
      if (modelPath.length <= 0) {
        println("Load model path not set, please check it.")
        System.exit(-1)
      }
      val predictData = sc.textFile(input, numDataPartitions)

      val optim = new AsyncAdam(stepSize)
      val model = new ParTorchModel(optim, torchModelPath)
      model.init()
      model.load(modelPath)
      val scores = predictData.mapPartitions {
        iterator =>
          iterator.sliding(batchSize, batchSize)
            .map(batch => model.predict(batch.toArray))
            .flatMap(f => f._1.zip(f._2))
            .map(f => (f._1, f._2.toDouble))
      }
      if (predictOutput.length > 0) {
        scores.saveAsTextFile(predictOutput)
      }
    }

    PSContext.stop()
    sc.stop()
  }

}
