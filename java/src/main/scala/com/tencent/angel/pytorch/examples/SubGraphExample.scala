package com.tencent.angel.pytorch.examples

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.pytorch.graph.subgraph.SubGraph
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.graph.utils.GraphIO
import org.apache.spark.{SparkConf, SparkContext}

object SubGraphExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgeInput", "")
    val featureInput = params.getOrElse("featureInput", "")
    val labelPath = params.getOrElse("labelPath", "")

    val edgeOutput = params.getOrElse("edgeOutput", "")
    val featureOutput = params.getOrElse("featureOutput", "")

    val numPartitions = params.getOrElse("numPartition", "10").toInt
    val psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")

    start()

    val sub = new SubGraph()
    sub.setInputFeaturePath(featureInput)
    sub.setOutputFeaturePath(featureInput)
    sub.setOutputFeaturePath(featureOutput)
    sub.setLabelPath(labelPath)
    sub.setUseBalancePartition(useBalancePartition)
    sub.setPartitionNum(numPartitions)
    sub.setPSPartitionNum(psNumPartition)
    sub.setStorageLevel(storageLevel)

    val edges = GraphIO.load(edgeInput, isWeighted = false)
    val edgeSamples = sub.transform(edges)
    GraphIO.save(edgeSamples, edgeOutput, seq = " ")

    stop()
  }

  def start(mode: String = "yarn-cluster"): Unit = {
    val conf = new SparkConf()
    conf.setMaster(mode)
    conf.setAppName("gcn")
    conf.set(AngelConf.ANGEL_PSAGENT_UPDATE_SPLIT_ADAPTION_ENABLE, "false")
    new SparkContext(conf)
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}
