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

import com.tencent.angel.pytorch.params.HasUseSecondOrder
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}

class DGI extends GNN with HasUseSecondOrder {

  override
  def makeGraph(edges: RDD[(Long, Long)], model: GNNPSModel): Dataset[_] = {
    val adj = edges.groupByKey($(partitionNum))

    if ($(useSecondOrder)) {
      // if second order is required, init neighbors on PS
      adj.mapPartitionsWithIndex((index, it) =>
        Iterator(GraphAdjPartition.apply(index, it)))
        .map(_.init(model, $(numBatchInit))).reduce(_ + _)
    }

    val dgiGraph = adj.mapPartitionsWithIndex((index, it) =>
      Iterator.single(GraphAdjPartition.apply(index, it).
        toMiniBatchDGIPartition(model, $(torchModelPath), $(useSecondOrder))))

    dgiGraph.persist($(storageLevel))
    dgiGraph.foreachPartition(_ => Unit)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[DGIPartition]
    SparkSession.builder().getOrCreate().createDataset(dgiGraph)
  }

  override
  def fit(model: GNNPSModel, graph: Dataset[_]): Unit = {
    val optim = getOptimizer

    for (curEpoch <- 1 to $(numEpoch)) {
      val (lossSum, totalTrain) = graph.rdd.map(_.asInstanceOf[DGIPartition]
        .trainEpoch(curEpoch, $(batchSize), model, $(featureDim), optim, $(numSamples)))
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
      println(s"curEpoch=$curEpoch train loss=${lossSum / totalTrain}")
    }

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
