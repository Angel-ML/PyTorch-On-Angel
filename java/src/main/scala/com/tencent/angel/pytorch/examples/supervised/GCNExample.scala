package com.tencent.angel.pytorch.examples.supervised

import com.tencent.angel.pytorch.graph.gcn.GCN
import com.tencent.angel.pytorch.graph.utils.GCNIO
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.graph.utils.GraphIO
import org.apache.spark.SparkContext

import scala.language.existentials

object GCNExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val featureInput = params.getOrElse("featurePath", "")
    val labelPath = params.getOrElse("labelPath", "")
    val predictOutputPath = params.getOrElse("predictOutputPath", "")
    val embeddingPath = params.getOrElse("embeddingPath", "")
    val batchSize = params.getOrElse("batchSize", "100").toInt
    val torchModelPath = params.getOrElse("torchModelPath", "model.pt")
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


    val gcn = new GCN()
    gcn.setTorchModelPath(torchModelPath)
    gcn.setFeatureDim(featureDim)
    gcn.setOptimizer(optimizer)
    gcn.setUseBalancePartition(false)
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

    val edges = GraphIO.load(edgeInput, isWeighted = false)
    val features = GCNIO.loadFeature(featureInput, sep = "\t")
    val labels = GCNIO.loadLabel(labelPath)

    val (model, graph) = gcn.initialize(edges, features, Option(labels))
    gcn.showSummary(model, graph)
    gcn.fit(model, graph)

    if (predictOutputPath.length > 0) {
      val predict = gcn.genLabels(model, graph)
      GraphIO.save(predict, predictOutputPath, seq = " ")
    }

    if (embeddingPath.length > 0) {
      val embedding = gcn.genEmbedding(model, graph)
      GraphIO.save(embedding, embeddingPath, seq = " ")
    }

    stop()
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}
