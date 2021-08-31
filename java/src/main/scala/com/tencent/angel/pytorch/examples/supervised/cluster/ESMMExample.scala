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

import com.tencent.angel.graph.utils.GraphIO
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.optim.OptimUtils
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{FloatType, StringType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

object ESMMExample {
  
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
    val numPartitions = params.getOrElse("numDataPartitions", "100").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "2").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "3").toInt
    val angelModelOutputPath = params.getOrElse("angelModelOutputPath", "")
    val angelModelInputPath = params.getOrElse("angelModelInputPath", "")
    val torchOutputModelPath = params.getOrElse("torchOutputModelPath", "")
    val rowType = params.getOrElse("rowType", "T_FLOAT_DENSE")
    val evals = params.getOrElse("evals", "auc")
    val storageLevel = params.getOrElse("storageLevel", "memory_only").toUpperCase()
    val shuffleInterval = params.getOrElse("shuffleInterval", "1000000").toInt
    val tdwColIndex = params.getOrElse("tdwColIndex", "0").toInt
    
    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)
    
    import com.tencent.angel.pytorch.model.ParTorchModel
    
    val conf = start(mode)
    
    // -javaagent:metricAgent.jar=useYarn=true for NGCP, which collects metrics
    if (conf.get("spark.hadoop.angel.metrics.enable", "false").toBoolean) {
      var executorJvmOptions = conf.get("spark.executor.extraJavaOptions")
      executorJvmOptions += " -javaagent:metricAgent.jar=useYarn=true "
      println(s"executorJvmOptions = $executorJvmOptions")
      conf.set("spark.executor.extraJavaOptions", executorJvmOptions)
    }
    
    val sc = SparkContext.getOrCreate()
    PSContext.getOrCreate(sc)
    
    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)
    
    val numDataPartitions = PartitionUtils.getDataPartitionNum(numPartitions, conf, numPartitionsFactor)
    println(s"numDataPartitions=$numDataPartitions")
    
    val trainInput = IOFunctions.loadString(trainPath, tdwColIndex).repartition(numDataPartitions) // training should be libsvm/libffm format
    if (actionType.equals("train")) {
      val rdd = trainInput.select("example").rdd.map(row => row.getString(0))
      val (train, test) = if (testRatio > 0) {
        val parts = rdd.randomSplit(Array(1 - testRatio, testRatio))
        (parts(0), parts(1))
      }
      else {
        (rdd, null)
      }
      
      train.persist(StorageLevel.fromString(storageLevel))
      println(s"train data count: ${train.count()}")
      if (testRatio > 0) {
        test.persist(StorageLevel.fromString(storageLevel))
        println(s"test data count: ${test.count()}")
      }
      
      val optim = OptimUtils.apply(optimizer, stepSize, decay)
      //      val optim = new AsyncAdam(stepSize)
      val model = new ParTorchModel(optim, torchModelPath)
      model.init()
      model.setMultiForwardOut(3)
      model.setUseAsync(async)
      
      for (epoch <- 1 to numEpoch) {
        val epochStartTime = System.currentTimeMillis()
        val (lossSum, size, maxBatch) = train.mapPartitions {
          iterator =>
            var (loss: Double, numBatch: Long, num: Long) = (0.0, 0L, 0L)
            val batchIterator = iterator.sliding(batchSize, batchSize)
            while (batchIterator.hasNext) {
              val batch = batchIterator.next().toArray
              loss += model.optimize(batch)
              numBatch += 1
              num += batch.length
            }
            Iterator((loss, num, numBatch))
        }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2, math.max(f1._3, f2._3)))
        print(s"epoch=$epoch loss=${lossSum / size} lr=${optim.getCurrentEta} ")
        // ctr:cvr:ctcvr
        val train_scores = train.mapPartitions { iterator =>
          iterator.toArray.sliding(batchSize, batchSize)
            .map(batch => model.predict(batch))
            .map(f => (f._1.map(_.split("#").map(_.toDouble)).flatMap(t => Iterator(t(0), t(1), t(0) * t(1))), f._2))
            .map(f => f._1.zip(f._2))
            .map { f =>
              val size = f.length / 3
              val ctr = new Array[(Double, Double)](size)
              val cvr = new Array[(Double, Double)](size)
              val ctcvr = new Array[(Double, Double)](size)
              
              for (i <- f.indices) {
                if (i % 3 == 0) {
                  ctr(i / 3) = (f(i)._1, f(i)._2.toDouble)
                }
                else if (i % 3 == 1) {
                  cvr(i / 3) = (f(i)._1, f(i)._2.toDouble)
                }
                else if (i % 3 == 2) {
                  ctcvr(i / 3) = (f(i)._1, f(i)._2.toDouble)
                }
              }
              (ctr, cvr, ctcvr)
            }
        }
        train_scores.persist(StorageLevel.fromString(storageLevel))
        train_scores.count()
        
        val train_ctr_scores = train_scores.flatMap(_._1)
        val train_cvr_scores = train_scores.flatMap(_._2)
        val train_ctcvr_scores = train_scores.flatMap(_._3)
        
        import com.tencent.angel.pytorch.eval.Evaluation
        val train_ctr_eval = Evaluation.eval(evals.split(","), train_ctr_scores)
        val train_cvr_eval = Evaluation.eval(evals.split(","), train_cvr_scores)
        val train_ctcvr_eval = Evaluation.eval(evals.split(","), train_ctcvr_scores)
        train_ctr_eval.foreach(f => print(s" train_ctr_${f._1}=${f._2}"))
        train_cvr_eval.foreach(f => print(s" train_cvr_${f._1}=${f._2}"))
        train_ctcvr_eval.foreach(f => print(s" train_ctcvr_${f._1}=${f._2}"))
        train_scores.unpersist(blocking = false)
        
        if (testRatio > 0) {
          // ctr:cvr:ctcvr
          val test_scores = test.mapPartitions { iterator =>
            iterator.toArray.sliding(batchSize, batchSize)
              .map(batch => model.predict(batch))
              .map(f => (f._1.map(_.split("#").map(_.toDouble)).flatMap(t => Iterator(t(0), t(1), t(0) * t(1))), f._2))
              .map(f => f._1.zip(f._2))
              .map { f =>
                val size = f.length / 3
                val ctr = new Array[(Double, Double)](size)
                val cvr = new Array[(Double, Double)](size)
                val ctcvr = new Array[(Double, Double)](size)
                
                for (i <- f.indices) {
                  if (i % 3 == 0) {
                    ctr(i / 3) = (f(i)._1, f(i)._2.toDouble)
                  }
                  else if (i % 3 == 1) {
                    cvr(i / 3) = (f(i)._1, f(i)._2.toDouble)
                  }
                  else if (i % 3 == 2) {
                    ctcvr(i / 3) = (f(i)._1, f(i)._2.toDouble)
                  }
                }
                (ctr, cvr, ctcvr)
              }
          }
          test_scores.persist(StorageLevel.fromString(storageLevel))
          test_scores.count()
          
          val test_ctr_scores = test_scores.flatMap(_._1)
          val test_cvr_scores = test_scores.flatMap(_._2)
          val test_ctcvr_scores = test_scores.flatMap(_._3)
          
          import com.tencent.angel.pytorch.eval.Evaluation
          val test_ctr_eval = Evaluation.eval(evals.split(","), test_ctr_scores)
          val test_cvr_eval = Evaluation.eval(evals.split(","), test_cvr_scores)
          val test_ctcvr_eval = Evaluation.eval(evals.split(","), test_ctcvr_scores)
          test_ctr_eval.foreach(f => print(s" test_ctr_${f._1}=${f._2}"))
          test_cvr_eval.foreach(f => print(s" test_cvr_${f._1}=${f._2}"))
          test_ctcvr_eval.foreach(f => print(s" test_ctcvr_${f._1}=${f._2}"))
          test_scores.unpersist(blocking = false)
        }
        val epochTime = System.currentTimeMillis() - epochStartTime
        print(s" time=${epochTime.toFloat / 1000}s")
        println()
        optim.step(maxBatch.toInt)
      }
      
      if (angelModelOutputPath.length > 0)
        model.save(angelModelOutputPath)
      if (torchOutputModelPath.length > 0)
        model.saveModule(torchOutputModelPath, SparkHadoopUtil.get.newConfiguration(sc.getConf))
    } else {
      assert(angelModelInputPath.length > 0 && predictOutputPath.length > 0,
        "Load model path or predict output path not set, please check it.")
      
      val predictData = trainInput.select("example").rdd.map(row => row.getString(0))
      
      val optim = OptimUtils.apply(optimizer, stepSize, decay)
      val model = new ParTorchModel(optim, torchModelPath)
      model.init()
      model.setMultiForwardOut(3)
      model.setUseAsync(async)
      model.load(angelModelInputPath)
      // target cvt ctr ctcvr
      val scores = predictData.mapPartitions {
        iterator =>
          iterator.sliding(batchSize, batchSize)
            .map(batch => model.predict(batch.toArray))
            .flatMap{ f =>
              f._1.indices.map(i =>
                Row.fromSeq(Seq[Any](f._1(i), f._2(i * 3), f._2(i * 3 + 1), f._2(i * 3 + 2))))
            }
      }
      val schema = StructType(Seq(
        StructField("target", StringType, nullable = false),
        StructField("ctr", FloatType, nullable = false),
        StructField("cvr", FloatType, nullable = false),
        StructField("ctcvr", FloatType, nullable = false)
      ))
      
      if (predictOutputPath.length > 0) {
        val results = trainInput.sparkSession.createDataFrame(scores, schema)
        GraphIO.save(results, predictOutputPath)
      }
    }
    
    stop()
  }
  def start(mode: String = "local"): SparkConf = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("ESMM")
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
