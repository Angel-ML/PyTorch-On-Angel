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

import com.tencent.angel.pytorch.eval.{AUC, Evaluation}
import com.tencent.angel.pytorch.params._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}


class GCN extends GNN with HasTestRatio {

  override
  def makeGraph(edges: DataFrame, model: GNNPSModel): Dataset[_] = {
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

    val gcnGraph = if (model.nnzTestLabels() == 0)
      adjGraph.map(_.toSemiGCNPartition(model, $(torchModelPath), $(testRatio)))
    else
      adjGraph.map(_.toSemiGCNPartition(model, $(torchModelPath)))

    // build GCN graph partitions
    //    val gcnGraph = adjGraph.map(_.toSemiGCNPartition(model, $(torchModelPath), $(testRatio)))
    gcnGraph.persist($(storageLevel))
    gcnGraph.count()
    adjGraph.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[GCNPartition]
    SparkSession.builder().getOrCreate().createDataset(gcnGraph)
  }

  def evaluate(model: GNNPSModel, graph: Dataset[_], isTest: Boolean = true): Map[String, Double] = {
    import com.tencent.angel.pytorch.eval.Evaluation._
    val scores = graph.rdd.flatMap(_.asInstanceOf[GCNPartition]
      .predictEpoch(0, $(batchSize) * 10, model,
        $(featureDim), $(numSamples), isTest)).flatMap(f => f._1.zip(f._2))
    Evaluation.eval(getEvaluations, scores)
  }


  override
  def fit(model: GNNPSModel, graph: Dataset[_], checkPointPath: String = null): Unit = {

    val optim = getOptimizer
    println(s"optimizer: $optim")
    println(s"evals: ${getEvaluations.mkString(",")}")

    val (trainSize, testSize) = graph.rdd.map(_.asInstanceOf[GCNPartition].getTrainTestSize)
      .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
    println(s"numTrain=$trainSize numTest=$testSize testRatio=${$(testRatio)} samples=${$(numSamples)}")

    val trainMetrics = evaluate(model, graph, false)
    val validateMetrics = evaluate(model, graph)
    print(s"curEpoch=0 ")
    trainMetrics.foreach(f => print(s"train_${f._1}=${f._2} "))
    validateMetrics.foreach(f => print(s"validate_${f._1}=${f._2} "))
    println()

    for (curEpoch <- 1 to $(numEpoch)) {
      val (lossSum, trainRight, numSteps) = graph.rdd.map(_.asInstanceOf[GCNPartition]
        .trainEpoch(curEpoch, $(batchSize), model,
          $(featureDim), optim, $(numSamples)))
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2, math.max(f1._3, f2._3)))

      // use max(steps) from all partition to forward the steps of optimizer
      optim.step(numSteps)

      val trainMetrics = evaluate(model, graph, false)
      val validateMetrics = evaluate(model, graph)

      print(s"curEpoch=$curEpoch train_loss=${lossSum / trainSize} ")
      trainMetrics.foreach(f => print(s"train_${f._1}=${f._2} "))
      validateMetrics.foreach(f => print(s"validate_${f._1}=${f._2} "))
      println()

      checkpointIfNeed(model, curEpoch)
      if (checkPointPath != null && checkPointPath.length > 0 && curEpoch % $(periods) == 0)
        save(model, checkPointPath, curEpoch)
    }
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
