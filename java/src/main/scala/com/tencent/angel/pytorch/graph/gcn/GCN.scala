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

import com.tencent.angel.pytorch.params._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}


class GCN extends GNN with HasTestRatio {

  override
  def makeGraph(edges: RDD[(Long, Long)], model: GNNPSModel): Dataset[_] = {
    // build adj graph partitions
    val adjGraph = edges.groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(GraphAdjPartition.apply(index, it)))

    adjGraph.persist($(storageLevel))
    adjGraph.foreachPartition(_ => Unit)
    adjGraph.map(_.init(model, $(numBatchInit))).reduce(_ + _)

    // build GCN graph partitions
    val gcnGraph = adjGraph.map(_.toSemiGCNPartition(model, $(torchModelPath), $(testRatio)))
    gcnGraph.persist($(storageLevel))
    gcnGraph.count()
    adjGraph.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[GCNPartition]
    SparkSession.builder().getOrCreate().createDataset(gcnGraph)
  }

  override
  def fit(model: GNNPSModel, graph: Dataset[_]): Unit = {

    val optim = getOptimizer

    val (trainSize, testSize) = graph.rdd.map(_.asInstanceOf[GCNPartition].getTrainTestSize)
      .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
    println(s"numTrain=$trainSize numTest=$testSize testRatio=${$(testRatio)} samples=${$(numSamples)}")

    for (curEpoch <- 1 to $(numEpoch)) {
      val (lossSum, trainRight) = graph.rdd.map(_.asInstanceOf[GCNPartition].trainEpoch(curEpoch, $(batchSize), model,
        $(featureDim), optim, $(numSamples))).reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
      val predRight = graph.rdd.map(_.asInstanceOf[GCNPartition].predictEpoch(curEpoch, $(batchSize) * 10, model,
        $(featureDim), $(numSamples))).reduce(_ + _)
      println(s"curEpoch=$curEpoch " +
        s"train loss=${lossSum / trainSize} " +
        s"train acc=${trainRight.toDouble / trainSize} " +
        s"test acc=${predRight.toDouble / testSize}")
    }
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
