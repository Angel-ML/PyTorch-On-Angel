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

import com.tencent.angel.pytorch.params.{HasTestRatio, HasWeighted}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

class DGI extends GNN with HasWeighted with HasTestRatio {

  override
  def makeGraph(edges: DataFrame, model: GNNPSModel, labelDF: Option[DataFrame],
                testLabelDF: Option[DataFrame], minId: Long, maxId: Long): Dataset[_] = {
    val adj = if ($(hasWeighted)) {
      edges.select("src", "dst", "weight").rdd
        .map(row => (row.getLong(0), (row.getLong(1), row.getFloat(2))))
        .filter(f => f._1 != f._2._1)
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjWeightedPartition.apply(index, it)))
    } else {
      edges.select("src", "dst").rdd
        .map(row => (row.getLong(0), row.getLong(1)))
        .filter(f => f._1 != f._2)
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator(GraphAdjPartition.apply(index, it)))
    }

    adj.persist($(storageLevel))
    adj.foreachPartition(_ => Unit)
    adj.map(_.init(model, $(numBatchInit))).reduce(_ + _)

    val dgiGraph = adj.map(_.toMiniBatchDGIPartition(model, $(torchModelPath), $(useSecondOrder), $(dataFormat)))

    dgiGraph.persist($(storageLevel))
    dgiGraph.foreachPartition(_ => Unit)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[DGIPartition]
    SparkSession.builder().getOrCreate().createDataset(dgiGraph)
  }

  override
  def fit(model: GNNPSModel, graph: Dataset[_], checkPointPath: String = null): Unit = {
    val optim = getOptimizer

    for (curEpoch <- 1 to $(numEpoch)) {
      val startTs = System.currentTimeMillis()
      val (lossSum, totalTrain, numSteps) = graph.rdd.map(_.asInstanceOf[DGIPartition]
        .trainEpoch(curEpoch, $(batchSize), model, $(featureDim), optim, $(numSamples), $(fieldNum), $(fieldMultiHot), $(testRatio)))
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2, math.max(f1._3, f2._3)))
      optim.step(numSteps)
      println(s"curEpoch=$curEpoch lr=${optim.getCurrentEta.toFloat} train loss=${lossSum / totalTrain} " +
        s"cost=${(System.currentTimeMillis() - startTs) / 1000.0f}s ")

      if (checkPointPath != null && checkPointPath.length > 0 && curEpoch % $(periods) == 0)
        save(model, checkPointPath, curEpoch)
    }

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
