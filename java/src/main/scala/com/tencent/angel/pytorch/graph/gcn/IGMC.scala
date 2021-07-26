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
import com.tencent.angel.pytorch.eval.Evaluation
import com.tencent.angel.pytorch.params.{HasTaskType, HasUseSharedSamples}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class IGMC extends BiGCN with HasTaskType with HasUseSharedSamples {

  override
  def makeGraph(edges: DataFrame, model: BiSAGEPSModel,
                labelDF: Option[DataFrame], testLabelDF: Option[DataFrame],
                userMinId: Long, userMaxId: Long): (Dataset[_], Dataset[_]) = {
    val adj_u = if ($(hasEdgeType)) {
      edges.select("src", "dst", "weight").rdd
        .map(row => (row.getLong(0), (row.getLong(1), row.getFloat(2).toInt, 0)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 0, $(hasEdgeType))))
    } else {
      throw new Exception("edge type has not found.")
    }

    val adj_i = if ($(hasEdgeType)) {
      edges.select("src", "dst", "weight").rdd
        .map(row => (row.getLong(1), (row.getLong(0), row.getFloat(2).toInt, 0)))
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjBiPartition.apply(index, it, 1, $(hasEdgeType))))
    } else {
      throw new Exception("edge type has not found.")
    }

    adj_u.persist($(storageLevel))
    adj_i.persist($(storageLevel))
    adj_u.foreachPartition(_ => Unit)
    adj_i.foreachPartition(_ => Unit)
    // init neighbors on PS
    adj_u.map(_.init(model, $(numBatchInit))).reduce(_ + _)
    adj_i.map(_.init(model, $(numBatchInit))).reduce(_ + _)

    val userGraph =
      adj_u.map(_.toIGMCPartition(model, $(torchModelPath), $(useSecondOrder), $(testRatio)))

    val itemGraph = adj_i.map(_.toIGMCPartition(model, $(torchModelPath), $(useSecondOrder), $(testRatio)))

    userGraph.persist($(storageLevel))
    userGraph.foreachPartition(_ => Unit)
    itemGraph.persist($(storageLevel))
    itemGraph.foreachPartition(_ => Unit)
    adj_u.unpersist(true)
    adj_i.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[IGMCPartition]
    (SparkSession.builder().getOrCreate().createDataset(userGraph),
      SparkSession.builder().getOrCreate().createDataset(itemGraph))
  }

  override
  def fit(model: BiSAGEPSModel, userGraph: Dataset[_], itemGraph: Dataset[_], checkPointPath: String): Unit = {
    val optim = getOptimizer
    println(s"optimizer: $optim")
    println(s"evals: ${getEvaluations.mkString(",")}")

    var startTs = System.currentTimeMillis()
    val (trainSize, testSize) = userGraph.rdd.map(_.asInstanceOf[IGMCPartition].getTrainTestSize)
      .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
    println(s"numTrain=$trainSize numTest=$testSize testRatio=${$(testRatio)} samples=${$(numSamples)}")

    val trainMetrics = evaluate(model, userGraph, false)
    val validateMetrics = evaluate(model, userGraph)
    print(s"curEpoch=0 ")
    trainMetrics.foreach(f => print(s"train_${f._1}=${f._2} "))
    validateMetrics.foreach(f => print(s"validate_${f._1}=${f._2} "))
    print(s"cost=${(System.currentTimeMillis() - startTs) / 1000}s ")
    println()

    for (curEpoch <- 1 to $(numEpoch)) {
      startTs = System.currentTimeMillis()

      val res = userGraph.rdd.map(_.asInstanceOf[IGMCPartition]
        .trainEpoch(curEpoch, $(batchSize), model, $(userFeatureDim), $(itemFeatureDim), optim, $(numSamples), 0, $(useSharedSamples)))
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
      print(s"cost=${(System.currentTimeMillis() - startTs) / 1000}s ")

      println()

      if (checkPointPath != null && checkPointPath.length > 0 && curEpoch % $(periods) == 0)
        save(model, checkPointPath, curEpoch)
    }

  }

  override
  def evaluate(model: BiSAGEPSModel, graph: Dataset[_], isTest: Boolean = true): Map[String, String] = {
    import com.tencent.angel.pytorch.eval.Evaluation._
    val scores = graph.rdd.flatMap(_.asInstanceOf[IGMCPartition]
      .predictEpoch(0, $(batchSize) * $(batchSizeMultiple), model,
        $(userFeatureDim), $(itemFeatureDim), $(numSamples), isTest)).flatMap(f => f._1.zip(f._2))
    Evaluation.eval(getEvaluations, scores).map(x => (x._1, x._2.toString))
  }

  override
  def evaluate(scores: RDD[(Float, Float)]): Map[String, String] = {
    import com.tencent.angel.pytorch.eval.Evaluation._
    Evaluation.eval(getEvaluations, scores).map(x => (x._1, x._2.toString))
  }

  def genEmbedding(model: BiSAGEPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[IGMCPartition]
      .genEmbedding($(batchSize) * $(batchSizeMultiple), model, $(userFeatureDim), $(itemFeatureDim), $(numSamples), graph.rdd.getNumPartitions, 0))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("src", LongType, nullable = false),
      StructField("dst", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }


  def genLabels(model: BiSAGEPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[IGMCPartition]
      .genLabel($(batchSize) * $(batchSizeMultiple), model, $(userFeatureDim), $(itemFeatureDim), $(numSamples), graph.rdd.getNumPartitions, 0, $(taskType)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3)))

    val schema = StructType(Seq(
      StructField("src", LongType, nullable = false),
      StructField("dst", LongType, nullable = false),
      StructField("label", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  def genLabelsEmbedding(model: BiSAGEPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[IGMCPartition]
      .genLabelsEmbedding($(batchSize) * $(batchSizeMultiple), model, $(userFeatureDim), $(itemFeatureDim), $(numSamples), graph.rdd.getNumPartitions, 0, $(taskType)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3, f._4)))

    val schema = StructType(Seq(
      StructField("src", LongType, nullable = false),
      StructField("dst", LongType, nullable = false),
      StructField("label", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }


}