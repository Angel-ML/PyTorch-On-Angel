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
package com.tencent.angel.pytorch.graph.gcn

import java.io.File

import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.io.DataLoaderUtils
import com.tencent.angel.pytorch.params._
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.graph.params._
import org.apache.hadoop.fs.permission.FsPermission
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkEnv}

class GNN(val uid: String) extends Serializable
  with HasTorchModelPath with HasBatchSize with HasFeatureDim
  with HasOptimizer with HasNumEpoch with HasNumSamples with HasNumBatchInit
  with HasPartitionNum with HasPSPartitionNum with HasUseBalancePartition
  with HasDataFormat with HasStorageLevel with  HasModelCheckPoint with HasPSModelCheckpoint
  with HasEvaluation {

  def this() = this(Identifiable.randomUID("GNN"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  def fit(model: GNNPSModel, graph: Dataset[_], checkPointPath: String = null): Unit = ???

  def initFeatures(model: GNNPSModel, features: Dataset[Row], minId: Long, maxId: Long): Unit = {
    features.rdd.filter(row => row.length > 0)
      .filter(row => row.get(0) != null)
      .map(row => (row.getLong(0), row.getString(1)))
      .filter(f => f._1 >= minId && f._1 <= maxId)
      .map(f => (f._1, SampleParser.parseFeature(f._2, $(featureDim), $(dataFormat))))
      .repartition($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(NodeFeaturePartition.apply(index, it)))
      .map(_.init(model, $(numBatchInit))).count()
  }

  def initLabels(model: GNNPSModel, labels: Dataset[Row], minId: Long, maxId: Long): Unit = {
    labels.rdd.map(row => (row.getLong(0), row.getFloat(1)))
      .filter(f => f._1 >= minId && f._1 <= maxId)
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(NodeLabelPartition.apply(index, it, model.dim)))
      .map(_.init(model)).count()
  }

  def initTestLabels(model: GNNPSModel, labels: Dataset[Row], minId: Long, maxId: Long): Unit = {
    labels.rdd.map(row => (row.getLong(0), row.getFloat(1)))
      .filter(f => f._1 >= minId && f._1 <= maxId)
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(NodeLabelPartition.apply(index, it, model.dim)))
      .map(_.initTestLabels(model)).count()
  }

  def getMinMaxId(edges: DataFrame): (Long, Long, Long) =
    edges.select("src", "dst").rdd
      .map(row => (row.getLong(0), row.getLong(1)))
      .filter(f => f._1 != f._2)
      .mapPartitions(DataLoaderUtils.summarizeApplyOp)
      .reduce(DataLoaderUtils.summarizeReduceOp)

  def getPartitionIndex(edges: DataFrame): RDD[Long] =
    edges.select("src", "dst").rdd
      .map(row => (row.getLong(0), row.getLong(1)))
      .filter(f => f._1 != f._2)
      .flatMap(f => Iterator(f._1, f._2))


  def makeGraph(edges: DataFrame, model: GNNPSModel): Dataset[_] = ???

  def initialize(edgeDF: DataFrame, featureDF: DataFrame): (GNNPSModel, Dataset[_]) =
    initialize(edgeDF, featureDF, None, None)

  def initialize(edgeDF: DataFrame,
                 featureDF: DataFrame,
                 labelDF: Option[DataFrame]): (GNNPSModel, Dataset[_]) =
    initialize(edgeDF, featureDF, labelDF, None)

  def initialize(edgeDF: DataFrame,
                 featureDF: DataFrame,
                 labelDF: Option[DataFrame],
                 testLabelDF: Option[DataFrame]): (GNNPSModel, Dataset[_]) = {
    val start = System.currentTimeMillis()

    // read edges
    edgeDF.persist(StorageLevel.DISK_ONLY)

    val (minId, maxId, numEdges) = getMinMaxId(edgeDF)
    println(s"minId=$minId maxId=$maxId numEdges=$numEdges")

    // create weights, graph on servers
    TorchModel.setPath($(torchModelPath))
    val torch = TorchModel.get()
    val weightsSize = torch.getParametersTotalSize
    println(s"weight total size=${weightsSize}")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    val model = GNNPSModel.apply(minId, maxId + 1, weightsSize, getOptimizer,
      getPartitionIndex(edgeDF), $(psPartitionNum), $(useBalancePartition))

    // initialize weights with torch values
    model.setWeights(torch.getParameters)
    TorchModel.put(torch)

    labelDF.foreach(f => initLabels(model, f, minId, maxId))
    testLabelDF.foreach(f => initTestLabels(model, f, minId, maxId))

    val graph = makeGraph(edgeDF, model)
    initFeatures(model, featureDF, minId, maxId)

    val end = System.currentTimeMillis()
    println(s"initialize cost ${(end - start) / 1000}s")
    (model, graph)
  }


  def genEmbedding(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[GNNPartition]
      .genEmbedding($(batchSize) * 10, model, $(featureDim), $(numSamples), graph.rdd.getNumPartitions))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  def genLabels(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[GNNPartition]
      .genLabels($(batchSize) * 10, model, $(featureDim), $(numSamples), graph.rdd.getNumPartitions))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", LongType, nullable = false),
      StructField("softmax", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  def genLabelsEmbedding(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[GNNPartition]
      .genLabelsEmbedding($(batchSize) * 10, model, $(featureDim), $(numSamples), graph.rdd.getNumPartitions))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3, f._4)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false),
      StructField("softmax", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }


  def showSummary(model: GNNPSModel, graph: Dataset[_]): Unit = {
    println(s"numNodesHasLabels=${model.nnzLabels()}")
    println(s"numTestLabels=${model.nnzTestLabels()}")
    println(s"nnzFeatures=${model.nnzFeatures()}")
    println(s"nnzNode=${model.nnzNodes()}")
    println(s"nnzEdge=${model.nnzEdge()}")
    println(s"nnzNeighbor=${model.nnzNeighbors()}")
    val numNodesHasOutEdges = graph.rdd.map(_.asInstanceOf[GNNPartition].numNodes).reduce(_ + _)
    val numEdges = graph.rdd.map(_.asInstanceOf[GNNPartition].numEdges).reduce(_ + _)
    println(s"numNodesWithOutEdges=$numNodesHasOutEdges")
    println(s"numEdges=$numEdges")

    val numNodesWithoutInDegree = graph.rdd.map(_.asInstanceOf[GNNPartition]
      .aloneNodes(model, graph.rdd.getNumPartitions).length).reduce(_ + _)
    println(s"numNodesWithoutInDegree=$numNodesWithoutInDegree")
  }

  def save(model: GNNPSModel, path: String): Unit = {
    // save pt to local file
    save(model, path, -1)
  }

  def save(model: GNNPSModel, path: String, epoch: Int): Unit = {

    def getDestinationPath: Path = {
      if (epoch < 0)
        new Path(path + "/" + $(torchModelPath))
      else
        new Path(path + "/" + s"${$(torchModelPath)}.${epoch}")
    }

    // save pt to local file with epoch_id
    val weights = model.readWeights()
    val torch = TorchModel.get()
    torch.gcnSave(s"gcn-train-temp-${epoch}.pt", weights)

    val hdfsPath = new Path(path)
    val conf = SparkHadoopUtil.get.newConfiguration(SparkEnv.get.conf)
    val fs = hdfsPath.getFileSystem(conf)
    val outputPath = new Path(fs.makeQualified(hdfsPath).toUri.getPath)
    if (!fs.exists(outputPath)) {
      val permission = new FsPermission(FsPermission.createImmutable(0x1ff.toShort))
      FileSystem.mkdirs(fs, outputPath, permission)
    }

    val file = new File(s"gcn-train-temp-${epoch}.pt")
    if (file.exists()) {
      val srcPath = new Path(file.getPath)
      val dstPath = getDestinationPath
      fs.copyFromLocalFile(srcPath, dstPath)
      println(s"save model to ${dstPath.toString}")
    }
    TorchModel.put(torch)
  }

  /**
    * Write checkpoint or result if need
    *
    * @param epoch
    */
  def checkpointIfNeed(model:GNNPSModel, epoch: Int): Unit = {
    var startTs = 0L
    if ($(checkpointInterval) > 0  && epoch % $(checkpointInterval) == 0 && epoch < $(numEpoch)) {
      println(s"Epoch=${epoch}, checkpoint the model")
      startTs = System.currentTimeMillis()
      model.checkpointMatrices(epoch)
      println(s"checkpoint use time=${System.currentTimeMillis() - startTs}ms")
    }
  }

}