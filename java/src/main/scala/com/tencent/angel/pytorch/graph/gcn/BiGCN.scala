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

import com.tencent.angel.pytorch.eval.{Evaluation, EvaluationM}
import com.tencent.angel.pytorch.params._
import org.apache.spark.rdd.RDD
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.context.PSContext
import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class BiGCN extends BiSAGE with HasTestRatio with HasValidate with HasNodeType with HasEdgeType
  with HasUseSharedSamples with HasNumLabels {

  override
  def initialize(edgeDF: DataFrame,
                 userFeatureDF: DataFrame,
                 itemFeatureDF: DataFrame,
                 labelDF: Option[DataFrame],
                 testLabelDF: Option[DataFrame]): (BiSAGEPSModel, Dataset[_], Dataset[_]) = {
    val start = System.currentTimeMillis()

    // read edges
    edgeDF.persist($(storageLevel))

    val (userMinId, userMaxId, numEdges1) = getMinMaxId(edgeDF, "src")
    println(s"userMinId=$userMinId userMaxId=$userMaxId numEdges=$numEdges1")
    val (itemMinId, itemMaxId, numEdges2) = getMinMaxId(edgeDF, "dst")
    println(s"itemMinId=$itemMinId itemMaxId=$itemMaxId numEdges=$numEdges2")

    // create weights, graph on servers
    TorchModel.setPath($(torchModelPath))
    val torch = TorchModel.get()
    if ($(userFieldNum) > 0) {
      assert(${userFeatureDim} == torch.getUserInputDim,
        s"userFeatureDim != torch.getUserInputDim: ${${userFeatureDim}} vs ${torch.getUserInputDim}")
      assert(${itemFeatureDim} == torch.getItemInputDim,
        s"itemFeatureDim != torch.getItemInputDim: ${${itemFeatureDim}} vs ${torch.getItemInputDim}")
    }

    val weightsSize = torch.getParametersTotalSize
    println(s"weight total size=$weightsSize")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    val model = if ($(userFieldNum) > 0)
      SparseBiSAGEPSModel.apply(userMinId, userMaxId + 1, itemMinId, itemMaxId + 1, weightsSize, getOptimizer,
        getPartitionIndex(edgeDF), $(psPartitionNum), $(useBalancePartition), $(userFeatEmbedDim),
        $(userFeatureDim), $(itemFeatEmbedDim), $(itemFeatureDim))
    else BiSAGEPSModel.apply(userMinId, userMaxId + 1, itemMinId, itemMaxId + 1, weightsSize, getOptimizer,
      getPartitionIndex(edgeDF), $(psPartitionNum), $(useBalancePartition))

    // initialize weights with torch values
    model.setWeights(torch.getParameters)
    // initialize embeddings
    if ($(userFieldNum) > 0) {
      if (${featEmbedPath}.length > 0) {
        println(s"load sparse feature embedding from ${${featEmbedPath}}.")
        model.asInstanceOf[SparseBiSAGEPSModel].loadFeatEmbed(${featEmbedPath})
      }
      else {
        println(s"init sparse feature embedding.")
        model.asInstanceOf[SparseBiSAGEPSModel].initUserEmbeddings()
        if (torch.getItemInputDim > 0)
          model.asInstanceOf[SparseBiSAGEPSModel].initItemEmbeddings()
      }
    }

    val (userGraph, itemGraph) = makeGraph(edgeDF, model, labelDF, testLabelDF, userMinId, userMaxId)
    // checkpoint while init neighbors
    model.checkpointMatrices(0)
    initFeatures(model, userFeatureDF, itemFeatureDF, userMinId, userMaxId, itemMinId, itemMaxId)

    // correct featureDim for sparse input after initFeatures
    if ($(userFieldNum) > 0) {
      setUserFeatureDim($(userFeatEmbedDim))
      setItemFeatureDim($(itemFeatEmbedDim))
    }
    TorchModel.put(torch)

    val end = System.currentTimeMillis()
    println(s"initialize cost ${(end - start) / 1000}s")
    val startTs = System.currentTimeMillis()
    if ($(saveCheckpoint))
      model.checkpointMatrices(1)
    println(s"Write checkpoint use time=${System.currentTimeMillis() - startTs}ms")
    (model, userGraph, itemGraph)
  }

  def makeGraph(edges: DataFrame, model: BiSAGEPSModel,
                labelDF: Option[DataFrame], testLabelDF: Option[DataFrame],
                userMinId: Long, userMaxId: Long): (Dataset[_], Dataset[_]) = {
    // 0 for init neighbors on userGraph
    val adj_u = if ($(hasNodeType) && $(hasEdgeType)) {
      edges.select("src", "dst", "edgeType", "dstType").rdd
        .map(row => (row.getLong(0), (row.getLong(1), row.getFloat(2).toInt, row.getFloat(3).toInt)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 0, $(hasEdgeType), $(hasNodeType))))
    } else if ($(hasNodeType)) {
      edges.select("src", "dst", "dstType").rdd
        .map(row => (row.getLong(0), (row.getLong(1), 0, row.getFloat(2).toInt)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 0, $(hasEdgeType), $(hasNodeType))))
    } else if ($(hasEdgeType)) {
      edges.select("src", "dst", "edgeType").rdd
        .map(row => (row.getLong(0), (row.getLong(1), row.getFloat(2).toInt, 0)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 0, $(hasEdgeType), $(hasNodeType))))
    } else {
      edges.select("src", "dst").rdd
        .map(row => (row.getLong(0), (row.getLong(1), 0, 0)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 0)))
    }

    // 1 for init neighbors on itemGraph
    val adj_i = if ($(hasNodeType) && $(hasEdgeType)) {
      edges.select("src", "dst", "edgeType", "dstType").rdd
        .map(row => (row.getLong(1), (row.getLong(0), row.getFloat(2).toInt, row.getFloat(3).toInt)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 1, $(hasEdgeType), $(hasNodeType))))
    } else if ($(hasNodeType)) {
      edges.select("src", "dst", "dstType").rdd
        .map(row => (row.getLong(1), (row.getLong(0), 0, row.getFloat(2).toInt)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 1, $(hasEdgeType), $(hasNodeType))))
    } else if ($(hasEdgeType)) {
      edges.select("src", "dst", "edgeType").rdd
        .map(row => (row.getLong(1), (row.getLong(0), row.getFloat(2).toInt, 0)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 1, $(hasEdgeType), $(hasNodeType))))
    } else {
      edges.select("src", "dst").rdd
        .map(row => (row.getLong(1), (row.getLong(0), 0, 0)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 1)))
    }

    adj_u.persist($(storageLevel))
    adj_i.persist($(storageLevel))
    adj_u.foreachPartition(_ => Unit)
    adj_i.foreachPartition(_ => Unit)
    // init neighbors on PS
    adj_u.map(_.init(model, $(numBatchInit))).reduce(_ + _)
    //retry for ps failed while init
    var nnzUserNode = model.nnzNodes(0)
    val numUserNodesHasOutEdges = adj_u.map(_.numNodes).reduce(_ + _)
    var retry = 0
    println("after init neighbors, nnzUserNode is: " + nnzUserNode + ", numUserNodesHasOutEdges is: "
      + numUserNodesHasOutEdges)
    while ((nnzUserNode != numUserNodesHasOutEdges) && retry < 2) {
      println("retry init neighbors on PS, nnzUserNode is: " + nnzUserNode + ", numUserNodesHasOutEdges is: "
        + numUserNodesHasOutEdges)
      adj_u.map(_.init(model, $(numBatchInit))).reduce(_ + _)
      nnzUserNode = model.nnzNodes(0)
      println("after retry init neighbors on PS, nnzUserNode is: " + nnzUserNode + ", numUserNodesHasOutEdges is: "
        + numUserNodesHasOutEdges)
      retry += 1
    }

    if (retry >= 2) {
      println("Retry limit reached, now exit.")
      System.exit(-1)
    }

    adj_i.map(_.init(model, $(numBatchInit))).reduce(_ + _)
    var nnzItemNode = model.nnzNodes(1)
    val numItemNodesHasOutEdges = adj_i.map(_.numNodes).reduce(_ + _)
    retry = 0
    println("after init neighbors, nnzItemNode is: " + nnzItemNode + ", numItemNodesHasOutEdges is: "
      + numItemNodesHasOutEdges)
    while ((nnzItemNode != numItemNodesHasOutEdges) && retry < 2) {
      println("retry init neighbors on PS, nnzItemNode is: " + nnzItemNode + ", numItemNodesHasOutEdges is: "
        + numItemNodesHasOutEdges)
      adj_i.map(_.init(model, $(numBatchInit))).reduce(_ + _)
      nnzItemNode = model.nnzNodes(1)
      println("after retry init neighbors on PS, nnzItemNode is: " + nnzItemNode + ", numItemNodesHasOutEdges is: "
        + numItemNodesHasOutEdges)
      retry += 1
    }

    if (retry >= 2) {
      println("Retry limit reached, now exit.")
      System.exit(-1)
    }

    if (${numLabels} == 1) {
      // init labels to labels and testLabels PSVectors
      labelDF.foreach(f => initLabels(model, f, userMinId, userMaxId))
      testLabelDF.foreach(f => initTestLabels(model, f, userMinId, userMaxId))
    } else {
      // init label arrays to userGraph PSMatrix
      labelDF.foreach(f => initMultiLabels(model, f, userMinId, userMaxId))
      testLabelDF.foreach(f => initMultiTestLabels(model, f, userMinId, userMaxId))
    }

    val userGraph = adj_u.map(_.toSemiBiGCNPartition(model, ${torchModelPath}, ${useSecondOrder}, ${testRatio}, ${numLabels}))
    val itemGraph = adj_i.map(_.toSemiBiGCNPartition(model, $(torchModelPath), $(useSecondOrder)))

    userGraph.persist($(storageLevel))
    userGraph.foreachPartition(_ => Unit)
    itemGraph.persist($(storageLevel))
    itemGraph.foreachPartition(_ => Unit)
    adj_u.unpersist(true)
    adj_i.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[BiGCNPartition]
    (SparkSession.builder().getOrCreate().createDataset(userGraph),
      SparkSession.builder().getOrCreate().createDataset(itemGraph))

  }

  override
  def fit(model: BiSAGEPSModel, userGraph: Dataset[_], itemGraph: Dataset[_], checkPointPath: String): Unit = {
    val optim = getOptimizer
    println(s"optimizer: $optim")
    println(s"evals: ${getEvaluations.mkString(",")}")

    var startTs = System.currentTimeMillis()
    val (trainSize, testSize) = userGraph.rdd.map(_.asInstanceOf[BiGCNPartition].getTrainTestSize())
      .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
    println(s"numTrain=$trainSize numTest=$testSize testRatio=${$(testRatio)} samples=${$(numSamples)}")
    // if testSize is not large, collect predict result to driver to calculate multi-label-auc
    if (testSize < 10000000 && ${evals}.startsWith("multi_auc")) {
      print(s"numTest=$testSize, set eval to ")
      setEvaluations("multi_auc_collect")
      println(s"${getEvaluations.mkString(",")}")
    }

    val trainMetrics = evaluate(model, userGraph, false)
    val validateMetrics = evaluate(model, userGraph)
    print(s"curEpoch=0 ")
    trainMetrics.foreach(f => print(s"train_${f._1}=${f._2} "))
    validateMetrics.foreach(f => print(s"validate_${f._1}=${f._2} "))
    print(s"cost=${(System.currentTimeMillis() - startTs) / 1000}s ")
    println()

    for (curEpoch <- 1 to $(numEpoch)) {
      startTs = System.currentTimeMillis()
      val res = userGraph.rdd.map(_.asInstanceOf[BiGCNPartition]
        .trainEpoch(curEpoch, $(batchSize), model, $(userFeatureDim), $(itemFeatureDim), optim, $(userNumSamples), $(itemNumSamples), 0, $(useSharedSamples), $(userFieldNum), $(itemFieldNum), $(fieldMultiHot)))
      res.persist($(storageLevel))

      val (lossSum, _, numSteps) = res.map(f => (f._1, f._2, f._3))
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2, math.max(f1._3, f2._3)))

      print(s"curEpoch=$curEpoch lr=${optim.getCurrentEta.toFloat} ")
      // use max(steps) from all partition to forward the steps of optimizer
      optim.step(numSteps)

      val trainMetrics = if ($(useSharedSamples)) evaluate(res.flatMap(f => f._4)) else evaluate(model, userGraph, false)
      print(s"train_loss=${lossSum / trainSize} ")
      trainMetrics.foreach(f => print(s"train_${f._1}=${f._2} "))
      if (curEpoch % $(validatePeriods) == 0) {
        val validateMetrics = evaluate(model, userGraph)
        validateMetrics.foreach(f => print(s"validate_${f._1}=${f._2} "))
      }
      print(s"cost=${(System.currentTimeMillis() - startTs) / 1000.0f}s ")

      println()

      if (checkPointPath != null && checkPointPath.length > 0 && curEpoch % $(periods) == 0) {
        save(model, checkPointPath, curEpoch)
        if (${dataFormat}.equals("sparse") && (${userFieldNum} > 0)) {
          saveFeatEmbed(model, checkPointPath, curEpoch)
        }
      }
    }

  }

  def evaluate(model: BiSAGEPSModel, graph: Dataset[_], isTest: Boolean = true): Map[String, String] = {
    import com.tencent.angel.pytorch.eval.Evaluation._
    val scores = graph.rdd.flatMap(_.asInstanceOf[BiGCNPartition]
      .predictEpoch(0, $(batchSize) *  $(batchSizeMultiple), model,
        $(userFeatureDim), $(itemFeatureDim), $(userNumSamples), $(itemNumSamples), isTest, $(userFieldNum), $(itemFieldNum), $(fieldMultiHot))).flatMap(f => f._1.zip(f._2))
      .persist(StorageLevel.MEMORY_ONLY)
    if (${numLabels} > 1) EvaluationM.eval(getEvaluations, scores, ${numLabels})
    else Evaluation.eval(getEvaluations, scores).map(x => (x._1, x._2.toString))
  }

  def evaluate(scores: RDD[(Float, Float)]): Map[String, String] = {
    import com.tencent.angel.pytorch.eval.Evaluation._
    if (${numLabels} > 1) EvaluationM.eval(getEvaluations, scores, ${numLabels})
    else Evaluation.eval(getEvaluations, scores).map(x => (x._1, x._2.toString))
  }

  override
  def genEmbedding(model: BiSAGEPSModel, graph: Dataset[_], graphType: Int): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[BiGCNPartition]
      .genEmbedding($(batchSize) * $(batchSizeMultiple), model, $(userFeatureDim), $(itemFeatureDim), $(userNumSamples), $(itemNumSamples),  graph.rdd.getNumPartitions, graphType, $(userFieldNum), $(itemFieldNum), $(fieldMultiHot)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  override
  def genLabelsEmbedding(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[BiGCNPartition]
      .genLabelsEmbedding($(batchSize) * $(batchSizeMultiple), model.asInstanceOf[BiSAGEPSModel],
        $(userFeatureDim), $(itemFeatureDim), $(userNumSamples), $(itemNumSamples), graph.rdd.getNumPartitions, 0, ${numLabels}, $(userFieldNum), $(itemFieldNum), $(fieldMultiHot)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3, f._4)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", StringType, nullable = false),
      StructField("embedding", StringType, nullable = false),
      StructField("softmax/sigmoid", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }
}