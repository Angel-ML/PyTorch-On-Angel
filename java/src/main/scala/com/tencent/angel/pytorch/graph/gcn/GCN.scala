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
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.storage.StorageLevel

class GCN extends GNN with HasTestRatio with HasValidate with HasUseSharedSamples {

  override
  def makeGraph(edges: DataFrame, model: GNNPSModel, labelDF: Option[DataFrame],
                testLabelDF: Option[DataFrame], minId: Long, maxId: Long): Dataset[_] = {
    // build adj graph partitions
    val adjGraph = edges.select("src", "dst").rdd
      .map(row => (row.getLong(0), row.getLong(1)))
      .filter(f => f._1 != f._2)
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(GraphAdjPartition.apply(index, it)))

    adjGraph.persist($(storageLevel))
    adjGraph.foreachPartition(_ => Unit)
    adjGraph.map(_.init(model, $(numBatchInit))).reduce(_ + _)

    if (${numLabels} == 1) {
      // init labels to labels and testLabels PSVectors
      labelDF.foreach(f => initLabels(model, f, minId, maxId))
      testLabelDF.foreach(f => initTestLabels(model, f, minId, maxId))
    } else {
      // init label arrays to userGraph PSMatrix
      labelDF.foreach(f => initMultiLabels(model, f, minId, maxId))
      testLabelDF.foreach(f => initMultiTestLabels(model, f, minId, maxId))
    }

    val gcnGraph = adjGraph.map(_.toSemiGCNPartition(model, $(torchModelPath), $(useSecondOrder),
      $(testRatio), $(numLabels)))

    // build GCN graph partitions
    //    val gcnGraph = adjGraph.map(_.toSemiGCNPartition(model, $(torchModelPath), $(testRatio)))
    gcnGraph.persist($(storageLevel))
    gcnGraph.count()
    adjGraph.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[GCNPartition]
    SparkSession.builder().getOrCreate().createDataset(gcnGraph)
  }

  override
  def fit(model: GNNPSModel, graph: Dataset[_], checkPointPath: String = null): Unit = {

    val optim = getOptimizer
    println(s"optimizer: $optim")
    println(s"evals: ${getEvaluations.mkString(",")}")

    var startTs = System.currentTimeMillis()
    val (trainSize, testSize) = graph.rdd.map(_.asInstanceOf[GCNPartition].getTrainTestSize())
      .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
    println(s"numTrain=$trainSize numTest=$testSize testRatio=${$(testRatio)} samples=${$(numSamples)}")

    val trainMetrics = evaluate(model, graph, false)
    val validateMetrics = evaluate(model, graph)
    print(s"curEpoch=0 ")
    trainMetrics.foreach(f => print(s"train_${f._1}=${f._2} "))
    validateMetrics.foreach(f => print(s"validate_${f._1}=${f._2} "))
    print(s"cost=${(System.currentTimeMillis() - startTs) / 1000}s ")
    println()

    for (curEpoch <- 1 to $(numEpoch)) {
      startTs = System.currentTimeMillis()
      val res = graph.rdd.map(_.asInstanceOf[GCNPartition]
        .trainEpoch(curEpoch, $(batchSize), model,
          $(featureDim), optim, $(numSamples), $(useSharedSamples), $(fieldNum), $(fieldMultiHot)))
      res.persist($(storageLevel))
      val (lossSum, trainRight, numSteps) = res.map(f => (f._1, f._2, f._3))
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2, math.max(f1._3, f2._3)))

      print(s"curEpoch=$curEpoch lr=${optim.getCurrentEta.toFloat} ")
      // use max(steps) from all partition to forward the steps of optimizer
      optim.step(numSteps)

      val trainMetrics = if ($(useSharedSamples)) evaluate(res.flatMap(f => f._4)) else evaluate(model, graph, false)
      print(s"train_loss=${lossSum / trainSize} ")
      trainMetrics.foreach(f => print(s"train_${f._1}=${f._2} "))
      if (curEpoch % $(validatePeriods) == 0) {
        val validateMetrics = evaluate(model, graph)
        validateMetrics.foreach(f => print(s"validate_${f._1}=${f._2} "))
      }
      print(s"cost=${(System.currentTimeMillis() - startTs) / 1000}s ")

      println()

      checkpointIfNeed(model, curEpoch)
      if (checkPointPath != null && checkPointPath.length > 0 && curEpoch % $(periods) == 0)
        save(model, checkPointPath, curEpoch)
    }
  }

  def evaluate(model: GNNPSModel, graph: Dataset[_], isTest: Boolean = true): Map[String, String] = {
    import com.tencent.angel.pytorch.eval.Evaluation._
    val scores = graph.rdd.flatMap(_.asInstanceOf[GCNPartition]
      .predictEpoch(0, $(batchSize) * $(batchSizeMultiple), model,
        $(featureDim), $(numSamples), isTest, $(fieldNum), $(fieldMultiHot))).flatMap(f => f._1.zip(f._2))
      .persist(StorageLevel.MEMORY_ONLY)
    if (${numLabels} > 1) EvaluationM.eval(getEvaluations, scores, ${numLabels})
    else Evaluation.eval(getEvaluations, scores).map(x => (x._1, x._2.toString))
  }

  def evaluate(scores: RDD[(Float, Float)]): Map[String, String] = {
    import com.tencent.angel.pytorch.eval.Evaluation._
    if (${numLabels} > 1) EvaluationM.eval(getEvaluations, scores, ${numLabels})
    else Evaluation.eval(getEvaluations, scores).map(x => (x._1, x._2.toString))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
