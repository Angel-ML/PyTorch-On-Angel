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

import com.tencent.angel.pytorch.model.ParTorchModel
import com.tencent.angel.pytorch.optim.AsyncSGD
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.core.metric.AUC
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LocalExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "")
    val batchSize = params.getOrElse("batchSize", "100").toInt
    val torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.1").toFloat
    val modelPath = params.getOrElse("modelPath", "")
    val modulePath = params.getOrElse("modulePath", "")
    val numEpoch = params.getOrElse("numEpoch", "10").toInt

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("local torch example")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    PSContext.getOrCreate(sc)
    val data = sc.textFile(input).repartition(10)

    val optim = new AsyncSGD(stepSize, 0.0)
    val model = new ParTorchModel(optim, torchModelPath)
    model.init()

    val parts = data.randomSplit(Array(0.8, 0.2))
    val (train, test) = (parts(0), parts(1))

    def evaluate(validate: RDD[String]): Double = {
      val scores = validate.mapPartitions {
        iterator =>
          iterator.sliding(batchSize, batchSize)
            .map(batch => model.predict(batch.toArray))
            .flatMap(f => f._1.zip(f._2))
            .map(f => (f._1.toDouble, f._2.toDouble))
      }
      new AUC().calculate(scores)
    }

    for (epoch <- 1 to numEpoch) {
      val epochStartTime = System.currentTimeMillis()
      val (lossSum, size) = train.mapPartitions {
        iterator =>
          iterator.sliding(batchSize, batchSize)
            .map(batch => (model.optimize(batch.toArray), batch.length))
      }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))

      val trainAuc = evaluate(train)
      val testAuc = evaluate(test)

      val epochTime = System.currentTimeMillis() - epochStartTime
      println(s"epoch=$epoch loss=${lossSum / size} trainAuc=$trainAuc testAuc=$testAuc time=${epochTime}ms")
    }

    if (modelPath.length > 0)
      model.save(modelPath)

    if (modulePath.length > 0)
      model.saveModule(modulePath, SparkHadoopUtil.get.newConfiguration(sc.getConf))

    PSContext.stop()
    sc.stop()
  }

}
