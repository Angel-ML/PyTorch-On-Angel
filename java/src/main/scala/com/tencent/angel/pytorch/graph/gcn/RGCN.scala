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

import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class RGCN extends GCN {

  override
  def makeGraph(edges: DataFrame, model: GNNPSModel, labelDF: Option[DataFrame],
                testLabelDF: Option[DataFrame], minId: Long, maxId: Long): Dataset[_] = {
    val adjGraph = edges.select("src", "dst", "type").rdd
      .map(row => (row.getLong(0), (row.getLong(1), row.getInt(2)))) // do not filter self-loop
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) => Iterator.single(GraphAdjTypePartition.apply(index, it)))

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

    // build R-GCN graph partition
    val rgcnGraph = adjGraph.map(_.toSemiRGCNPartition(model, $(torchModelPath), $(useSecondOrder),
      $(testRatio), $(numLabels)))
    rgcnGraph.persist($(storageLevel))
    rgcnGraph.count()
    adjGraph.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[RGCNPartition]
    SparkSession.builder().getOrCreate().createDataset(rgcnGraph)
  }

  override
  def genEmbedding(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[GNNPartition]
      .genEmbedding($(batchSize) * 10, model, $(featureDim), $(numSamples), graph.rdd.getNumPartitions, false, $(fieldNum), $(fieldMultiHot)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  override
  def genLabels(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[GNNPartition]
      .genLabels($(batchSize) * 10, model, $(featureDim), $(numSamples), graph.rdd.getNumPartitions,
        $(numLabels), false, $(fieldNum), $(fieldMultiHot)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", StringType, nullable = false),
      StructField("softmax", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

}