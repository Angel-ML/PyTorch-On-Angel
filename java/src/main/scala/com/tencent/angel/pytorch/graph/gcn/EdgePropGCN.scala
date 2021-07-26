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

import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.params.HasEdgeFeatureDim
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

class EdgePropGCN extends GCN with HasEdgeFeatureDim {

  override
  def makeGraph(edges: DataFrame, model: GNNPSModel, labelDF: Option[DataFrame],
                testLabelDF: Option[DataFrame], minId: Long, maxId: Long): Dataset[_] = {
    // build adj graph partitions
    val adjGraph = edges.select("src", "dst", "feature").rdd
      .map(row => (row.getLong(0), row.getLong(1), row.getString(2)))
      .filter(f => f._1 != f._2)
      .map(f => (f._1, (f._2, SampleParser.parseEdgeFeature(f._3, $(edgeFeatureDim), $(dataFormat)))))
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(GraphAdjEdgePartition.apply(index, it)))

    adjGraph.persist($(storageLevel))
    adjGraph.foreachPartition(_ => Unit)
    if ($(useSecondOrder))
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

    val gcnGraph =
      adjGraph.map(_.toSemiGCNPartition(model, $(torchModelPath), $(useSecondOrder), $(testRatio), $(numLabels)))

    // build GCN graph partitions
    //    val gcnGraph = adjGraph.map(_.toSemiGCNPartition(model, $(torchModelPath), $(testRatio)))
    gcnGraph.persist($(storageLevel))
    gcnGraph.count()
    adjGraph.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[EdgePropGCNPartition]
    SparkSession.builder().getOrCreate().createDataset(gcnGraph)
  }

}