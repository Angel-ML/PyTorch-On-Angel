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
package com.tencent.angel.pytorch.graph.gcn.hetAttention

import com.tencent.angel.pytorch.graph.gcn._
import com.tencent.angel.pytorch.params.{HasNodeType, HasUserFeatureDim}
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class HAN extends GCN with HasNodeType with HasUserFeatureDim {

  var itemTypes: Int = _
  def setItemTypes(a: Int): Unit = { this.itemTypes = a }

  override def makeGraph(edges: DataFrame, model: GNNPSModel,
                         labelDF: Option[DataFrame], testLabelDF: Option[DataFrame],
                         userMinId: Long, userMaxId: Long): Dataset[_] = {
    val adjWorker = edges.select("src", "dst", "weight").rdd
      .map(row => (row.getLong(0), (row.getLong(1), row.getFloat(2).toInt)))
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) => Iterator.single(GraphAdjTypePartition.apply(index, it)))

    val adjPS = edges.select("src", "dst", "weight").rdd
      .map(row => (row.getLong(1), (row.getLong(0), row.getFloat(2).toInt)))
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) => Iterator.single(GraphAdjTypePartition.apply(index, it)))

    adjWorker.persist($(storageLevel))
    adjPS.persist($(storageLevel))
    adjWorker.foreachPartition(_ => Unit)
    adjPS.foreachPartition(_ => Unit)
    // init item-user neighborTable to ps
    adjPS.map(_.init(model, $(numBatchInit))).reduce(_ + _)
    adjPS.unpersist(true)

    if (${numLabels} == 1) {
      // init labels to labels and testLabels PSVectors
      labelDF.foreach(f => initLabels(model, f, userMinId, userMaxId))
      testLabelDF.foreach(f => initTestLabels(model, f, userMinId, userMaxId))
    } else {
      // init label arrays to userGraph PSMatrix
      labelDF.foreach(f => initMultiLabels(model, f, userMinId, userMaxId))
      testLabelDF.foreach(f => initMultiTestLabels(model, f, userMinId, userMaxId))
    }

    // create user-item graph for workers (only support semi-supervised currently)
    val userGraph = adjWorker.map(_.toSemiHANPartition(model, $(torchModelPath), $(useSecondOrder),
      $(testRatio), itemTypes, $(numLabels)))
    userGraph.persist(${storageLevel})
    userGraph.foreachPartition(_ => Unit)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[HANPartition]
    SparkSession.builder().getOrCreate().createDataset(userGraph)
  }

  override
  def genEmbedding(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[HANPartition]
      .genEmbedding($(batchSize) * $(batchSizeMultiple), model, $(featureDim), $(numSamples),
        graph.rdd.getNumPartitions, false, $(fieldNum), $(fieldMultiHot)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  override def genLabelsEmbedding(model: GNNPSModel, graph: Dataset[_]): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[HANPartition]
      .genLabelsEmbedding($(batchSize) * $(batchSizeMultiple), model,  $(featureDim), $(numSamples),
        graph.rdd.getNumPartitions, $(numLabels), false, $(fieldNum), $(fieldMultiHot)))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2, f._3, f._4)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", StringType, nullable = false),
      StructField("embedding", StringType, nullable = false),
      StructField("softmax", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }
}