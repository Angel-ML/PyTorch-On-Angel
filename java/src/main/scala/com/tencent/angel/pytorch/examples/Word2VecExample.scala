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

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ps.storage.matrix.PartitionSourceArray
import com.tencent.angel.pytorch.embedding.{Word2VecModel, Word2VecParam}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.feature.{Features, SubSampling}
import com.tencent.angel.spark.ml.util.SparkUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

object Word2VecExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "")
    val output = params.getOrElse("output", "")
    val loadPath = params.getOrElse("loadPath", "")
    val torchModelPath = params.getOrElse("torchModelPath", "")
    val embeddingDim = params.getOrElse("embedding", "32").toInt
    val windowSize = params.getOrElse("window", "10").toInt
    val numNegSamples = params.getOrElse("negative", "5").toInt
    val numEpoch = params.getOrElse("epoch", "5").toInt
    val stepSize = params.getOrElse("stepSize", "0.1").toFloat
    val decayRate = params.getOrElse("decayRate", "0.5").toFloat
    val batchSize = params.getOrElse("batchSize", "50").toInt
    val numPartitions = params.getOrElse("numParts", "10").toInt
    val withSubSample = params.getOrElse("subSample", "true").toBoolean
    val withRemapping = params.getOrElse("remapping", "true").toBoolean
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val numExecutorDataPartitions = params.getOrElse("numDataPartitions", "100").toInt
    val saveModelInterval = params.getOrElse("saveModelInterval", "2").toInt
    val checkpointInterval = params.getOrElse("checkpointInterval", "5").toInt

    val conf = new SparkConf().setAppName("Word2Vec on PyTONA")
    //conf.setMaster("local[1]")
    conf.set(AngelConf.ANGEL_PS_PARTITION_SOURCE_CLASS, classOf[PartitionSourceArray].getName)
    conf.set(AngelConf.ANGEL_PS_BACKUP_MATRICES, "")
    conf.set(AngelConf.ANGEL_PS_BACKUP_INTERVAL_MS, "100000000")
    conf.set("io.file.buffer.size", "16000000")
    conf.set("spark.hadoop.io.file.buffer.size", "16000000")
    val executorJvmOptions = " -verbose:gc -XX:-PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:<LOG_DIR>/gc.log " +
      "-XX:+UseG1GC -XX:MaxGCPauseMillis=1000 -XX:G1HeapRegionSize=32M " +
      "-XX:InitiatingHeapOccupancyPercent=50 -XX:ConcGCThreads=4 -XX:ParallelGCThreads=4 "
    println(s"executorJvmOptions = $executorJvmOptions")
    conf.set("spark.executor.extraJavaOptions", executorJvmOptions)
    val sc = new SparkContext(conf)
    //sc.setLogLevel("ERROR")
    val numCores = SparkUtils.getNumCores(conf)
    // The number of partition is more than the cores. We do this to achieve dynamic load balance.
    var numDataPartitions = (numCores * 3.0).toInt
    if(numExecutorDataPartitions > numDataPartitions) {
      numDataPartitions = numExecutorDataPartitions
    }
    println(s"numDataPartitions=$numDataPartitions")

    val data = sc.textFile(input)
    var corpus: RDD[Array[Int]] = null
    var denseToString: Option[RDD[(Int, String)]] = None
    if (withRemapping) {
      val temp = Features.corpusStringToInt(data)
      corpus = temp._1
      denseToString = Some(temp._2)
    } else {
      corpus = Features.corpusStringToIntWithoutRemapping(data)
    }
    //Subsample will use ps, so start ps before subsample
    PSContext.getOrCreate(sc)

    val (maxWordId, docs) = if (withSubSample) {
      corpus.persist(StorageLevel.DISK_ONLY)
      val subsampleTmp = SubSampling.sampling(corpus)
      (subsampleTmp._1, subsampleTmp._2.repartition(numDataPartitions))
    } else {
      val tmp = corpus.repartition(numDataPartitions)
      (tmp.map(_.max).max().toLong + 1, tmp)
    }
    docs.persist(StorageLevel.fromString(storageLevel))
    val numDocs = docs.count()
    val numTokens = docs.map(_.length).sum().toLong
    val maxLength = docs.map(_.length).max()
    println(s"numDocs=$numDocs maxWordId=$maxWordId numTokens=$numTokens maxLength=$maxLength")

    val param = new Word2VecParam()
      .setTorchModelPath(torchModelPath)
      .setLearningRate(stepSize)
      .setDecayRate(decayRate)
      .setEmbeddingDim(embeddingDim)
      .setBatchSize(batchSize)
      .setWindowSize(windowSize)
      .setNumPSPart(Some(numPartitions))
      .setSeed(Random.nextInt())
      .setNumEpoch(numEpoch)
      .setNegSample(numNegSamples)
      .setMaxIndex(maxWordId)
      .setNumRowDataSet(numDocs)
      .setMaxLength(maxLength)
      .setModelCPInterval(checkpointInterval)
      .setModelSaveInterval(saveModelInterval)
    val model = new Word2VecModel(param)
    if(loadPath.length > 0) {
      model.load(loadPath)
    } else  {
      model.randomInitialize(Random.nextInt())
    }
    model.train(docs, param, output)
    model.save(output)
    denseToString.foreach(rdd => rdd.map(f => s"${f._1}:${f._2}").saveAsTextFile(output + "/mapping"))
    stop()
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }
}
