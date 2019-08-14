package com.tencent.angel.pytorch.examples.unsupervised

import com.tencent.angel.pytorch.graph.gcn.DGI
import com.tencent.angel.pytorch.graph.utils.GCNIO
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.graph.utils.GraphIO
import org.apache.spark.SparkContext

import scala.language.existentials

object DGIExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInput = params.getOrElse("edgePath", "")
    val featureInput = params.getOrElse("featurePath", "")
    val embeddingPath = params.getOrElse("embeddingPath", "")
    val batchSize = params.getOrElse("batchSize", "100").toInt
    val torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.01").toDouble
    val featureDim = params.getOrElse("featureDim", "-1").toInt
    val optimizer = params.getOrElse("optimizer", "adam")
    val psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    val numPartitions = params.getOrElse("numPartitions", "5").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val numEpoch = params.getOrElse("numEpoch", "10").toInt
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val second = params.getOrElse("second", "false").toBoolean
    val format = params.getOrElse("format", "sparse")

    val dgi = new DGI()
    dgi.setTorchModelPath(torchModelPath)
    dgi.setFeatureDim(featureDim)
    dgi.setOptimizer(optimizer)
    dgi.setUseBalancePartition(false)
    dgi.setBatchSize(batchSize)
    dgi.setStepSize(stepSize)
    dgi.setPSPartitionNum(psNumPartition)
    dgi.setPartitionNum(numPartitions)
    dgi.setUseBalancePartition(useBalancePartition)
    dgi.setNumEpoch(numEpoch)
    dgi.setStorageLevel(storageLevel)
    dgi.setUseSecondOrder(second)
    dgi.setDataFormat(format)

    val edges = GraphIO.load(edgeInput, isWeighted = false)
    val features = GCNIO.loadFeature(featureInput, sep = "\t")

    val (model, graph) = dgi.initialize(edges, features)
    dgi.showSummary(model, graph)
    dgi.fit(model, graph)

    if (embeddingPath.length > 0) {
      val embedding = dgi.genEmbedding(model, graph)
      GraphIO.save(embedding, embeddingPath, seq = " ")
    }

    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }

}
