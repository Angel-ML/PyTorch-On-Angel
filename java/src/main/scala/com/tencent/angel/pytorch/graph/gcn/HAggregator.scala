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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import com.tencent.angel.pytorch.params.{HasUseWeightedAggregate, HasWeighted}

class HAggregator extends RGCN with HasWeighted with HasUseWeightedAggregate{

  override
  def makeGraph(edges: DataFrame, model: GNNPSModel, labelDF: Option[DataFrame],
                testLabelDF: Option[DataFrame], minId: Long, maxId: Long): Dataset[_] = {
    val adjGraph = makePartition(edges, 0, 1)

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
    val rgcnGraph = adjGraph.map(_.toSemiHAggregatorPartition(model, $(torchModelPath), $(useSecondOrder),
      $(testRatio), $(numLabels), false))

    rgcnGraph.persist($(storageLevel))
    rgcnGraph.count()
    adjGraph.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[HAggregatorPartition]
    SparkSession.builder().getOrCreate().createDataset(rgcnGraph)
  }

  def makePartition(edges: DataFrame, start: Int = 0, end: Int = 1): RDD[GraphAdjTypePartition] = {
    if ($(hasWeighted)) {
      edges.select("src", "dst", "weight", "type").rdd
        .map(row => (row.getLong(start), (row.getLong(end), row.getFloat(2), row.getInt(3)))) // do not filter self-loop
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) => Iterator.single(GraphAdjWeightedTypePartition.apply(index, it, $(hasWeighted), $(hasUseWeightedAggregate))))
    } else {
      edges.select("src", "dst", "type").rdd
        .map(row => (row.getLong(start), (row.getLong(end), row.getInt(2)))) // do not filter self-loop
        .groupByKey($(partitionNum))
        .mapPartitionsWithIndex((index, it) => Iterator.single(GraphAdjTypePartition.apply(index, it)))
    }
  }

  def genEmbedding(model: GNNPSModel, graph: Dataset[_], cur_metapath: Integer, aggregator_in_scala: Boolean = false): DataFrame = {
    val ret = graph.rdd.flatMap(_.asInstanceOf[HAggregatorPartition]
      .genEmbedding($(batchSize) * 10, model, $(featureDim), $(numSamples), graph.rdd.getNumPartitions,
        false, $(fieldNum), $(fieldMultiHot), cur_metapath, $(hasUseWeightedAggregate), aggregator_in_scala))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  def concatFeatures(featuresDF: DataFrame, dims: Array[(Int, Int)]): (DataFrame, Int) = {
    // create initial 0 vectors for each type
    val type_vector_array = dims.map(r => (r._1, new Array[Int](r._2).mkString(" ")))

    // concat with other types of 0 vectors
    var temp_df = featuresDF
      .select("node", "type", "feature")
      .rdd
      .map(row => (row.getLong(0), getConcatFeatures(row.getInt(1), row.getString(2), type_vector_array)))
    val dim = temp_df.first()._2.split(" ").length

    val spark = SparkSession.builder.getOrCreate()
    import spark.implicits._
    (temp_df.toDF("node", "feature"), dim)
  }

  def getConcatFeatures(nodeType: Integer, feature: String, type_len_array: Array[(Int, String)]): String ={
    var s : String = ""
    var first: Boolean = true
    for ((t, vec_str) <- type_len_array){
      if (!first){
        s += " "
      }
      if (nodeType == t){
        s += feature
      }else{
        s += vec_str
      }
      first = false
    }
    s
  }

  def recodeNodes(featureDFs: Map[String, DataFrame]): (Map[String, Map[Long, Long]], Map[String, Int]) ={
    val nodeType2id = featureDFs.keys.zipWithIndex.toMap
    var nodeId = 0
    var nodeType: Map[Long, Integer] = Map()
    val node2idMap = featureDFs.map{ case (name, df) =>
      val idMap = df.select("node").rdd.map(r => (r.getLong(0), r.getLong(0) + nodeId)).collect().toMap
      idMap.values.foreach(r => nodeType += (r -> nodeType2id(name)))
      nodeId += df.count().toInt
      (name, idMap)
    }.toMap

    (node2idMap, nodeType2id)
  }

  def flattenFeatures(featureDFs: Map[String, DataFrame], node2idMap: Map[String, Map[Long, Long]], nodeType2id: Map[String, Int]): DataFrame ={
    val mapIdFeatures = featureDFs.map{ case (name, df) =>
      df.select("node", "feature").rdd.map(r => (node2idMap(name)(r.getLong(0)), nodeType2id(name), r.getString(1)))
    }.reduce(_.union(_))
    val spark = SparkSession.builder.getOrCreate()
    import spark.implicits._

    mapIdFeatures.toDF("node", "type", "feature")
  }

  def flattenEdges(edgeDFs: Map[String, DataFrame], node2idMap: Map[String, Map[Long, Long]]): (DataFrame, Map[String, Int]) = {
    val edgeType2id = edgeDFs.keys.flatMap { name =>
      val keys = name.split("-") // split u-i into u, i
      Array(name, keys.reverse.mkString("-")).distinct
    }.zipWithIndex.toMap
    val spark = SparkSession.builder.getOrCreate()
    import spark.implicits._
    if ($(hasWeighted)) {
      val mapIdEdges = edgeDFs.map { case (name, df) =>
        val keys = name.split("-") // split u-i into u, i
        df.select("src", "dst", "weight").rdd.map(r => (node2idMap(keys(0))(r.getLong(0)), node2idMap(keys(1))(r.getLong(1)), r.getFloat(2), edgeType2id(name)))
          .union(df.select("dst", "src", "weight").rdd.map(r => (node2idMap(keys(1))(r.getLong(0)), node2idMap(keys(0))(r.getLong(1)), r.getFloat(2), edgeType2id(keys.reverse.mkString("-")))))
      }.reduce(_.union(_))
      (mapIdEdges.toDF("src", "dst", "weight", "type"), edgeType2id)
    } else {
      val mapIdEdges = edgeDFs.map { case (name, df) =>
        val keys = name.split("-") // split u-i into u, i
        df.select("src", "dst").rdd.map(r => (node2idMap(keys(0))(r.getLong(0)), node2idMap(keys(1))(r.getLong(1)), edgeType2id(name)))
          .union(df.select("dst", "src").rdd.map(r => (node2idMap(keys(1))(r.getLong(0)), node2idMap(keys(0))(r.getLong(1)), edgeType2id(keys.reverse.mkString("-")))))
      }.reduce(_.union(_))
      (mapIdEdges.toDF("src", "dst", "type"), edgeType2id)
    }
  }

  def getNodeTypeFeatures(featuresDF: DataFrame, node2idMap: Map[String, Map[Long, Long]], nodeTypeStr: String): DataFrame = {
    val id2nodeMap = node2idMap.map(r => (r._1, r._2.map(node => (node._2, node._1)).toMap)).toMap
    val filterFeaturesDF = featuresDF.select("node", "feature").rdd
      .filter(r => id2nodeMap(nodeTypeStr).contains(r.getLong(0)))
      .map(r => (id2nodeMap(nodeTypeStr)(r.getLong(0)), r.getString(1)))
    val spark = SparkSession.builder.getOrCreate()
    import spark.implicits._
    filterFeaturesDF.toDF("node", "feature")
  }
}
