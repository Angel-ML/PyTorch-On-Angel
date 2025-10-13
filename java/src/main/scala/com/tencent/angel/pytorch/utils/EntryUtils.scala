package com.tencent.angel.pytorch.utils

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.spark.context.PSContext
import org.apache.spark.{SparkConf, SparkContext}

object EntryUtils {
  def start(mode: String = "local", torchModelPath: String, name: String = "pytorch on angel",
            actionType: String = "train", optimizer: String = "adam"): SparkConf = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName(name)
    conf.set("spark.executor.extraLibraryPath", "./torch/torch-lib")
    conf.set("spark.executorEnv.OMP_NUM_THREADS", "2")
    conf.set("spark.executorEnv.MKL_NUM_THREADS", "2")

    conf.set("spark.hadoop.angel.embedding.optimizer", optimizer)
    // angel.embedding.normalized
    if (actionType == "train" && conf.get("spark.hadoop.angel.embedding.normalized",
      "true").toBoolean) {
      conf.set("spark.hadoop.angel.embedding.normalized", "true")
    } else {
      conf.set("spark.hadoop.angel.embedding.normalized", "false")
    }

    val sc = new SparkContext(conf)
    sc.addFile(torchModelPath)
    if (sc.isLocal)
      sc.setLogLevel("ERROR")
    conf
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }
}
