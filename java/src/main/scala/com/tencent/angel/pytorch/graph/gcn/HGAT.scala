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

import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

class HGAT extends BiSAGE {

  override
  def makeGraph(edges: DataFrame, model: BiSAGEPSModel): (Dataset[_], Dataset[_]) = {
    val adj_u = edges.select("src", "dst").rdd
      .map(row => (row.getLong(0), (row.getLong(1), 0, 0)))
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator(GraphAdjBiPartition.apply(index, it, 0))) // 0 for init neighbors on userGraph

    val adj_i = edges.select("src", "dst").rdd
      .map(row => (row.getLong(1), (row.getLong(0), 0, 0)))
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator(GraphAdjBiPartition.apply(index, it, 1))) // 1 for init neighbors on itemGraph

    adj_u.persist($(storageLevel))
    adj_i.persist($(storageLevel))
    adj_u.foreachPartition(_ => Unit)
    adj_i.foreachPartition(_ => Unit)
    // init neighbors on PS
    adj_u.map(_.init(model, $(numBatchInit))).reduce(_ + _)
    adj_i.map(_.init(model, $(numBatchInit))).reduce(_ + _)

    val userGraph = adj_u.map(_.toHGATPartition(model, $(torchModelPath), $(useSecondOrder), $(dataFormat)))

    val itemGraph = adj_i.map(_.toHGATPartition(model, $(torchModelPath), $(useSecondOrder), $(dataFormat)))

    userGraph.persist($(storageLevel))
    userGraph.foreachPartition(_ => Unit)
    itemGraph.persist($(storageLevel))
    itemGraph.foreachPartition(_ => Unit)
    adj_u.unpersist(true)
    adj_i.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[HGATPartition]
    (SparkSession.builder().getOrCreate().createDataset(userGraph),
      SparkSession.builder().getOrCreate().createDataset(itemGraph))

  }

  override
  def fit(model: BiSAGEPSModel, userGraph: Dataset[_], itemGraph: Dataset[_], checkPointPath: String): Unit = {
    val optim = getOptimizer

    for (curEpoch <- 1 to $(numEpoch)) {
      val start = System.currentTimeMillis()
      val (lossSum1, totalTrain1, numSteps1) = userGraph.rdd.map(_.asInstanceOf[HGATPartition]
        .trainEpoch(curEpoch, $(batchSize), model, $(userFeatureDim), $(itemFeatureDim), optim, $(userNumSamples), $(itemNumSamples), 0, $(userFieldNum), $(itemFieldNum), $(fieldMultiHot), $(testRatio)))
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2, math.max(f1._3, f2._3)))

      optim.step(numSteps1)
      println(s"curEpoch=$curEpoch lr=${optim.getCurrentEta.toFloat} train loss=${lossSum1 / totalTrain1}, cost: ${(System.currentTimeMillis() - start) / 1000.0f} s")

      if (checkPointPath != null && checkPointPath.length > 0 && curEpoch % $(periods) == 0)
        save(model, checkPointPath, curEpoch)
    }
  }
}