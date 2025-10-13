package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.pytorch.data.SampleParser
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class Aggregator extends GCN {

  def getConcatFeatures(model: GNNPSModel,
                        graph: Dataset[_],
                        edges: DataFrame,
                        features: DataFrame,
                        hops: Int): DataFrame = {
    val (minId, maxId, _) = getMinMaxId(edges)
    var featureList = features
    var feats: DataFrame = null
    for (hop <- 1 to hops) {
      initFeatures(model, features, minId, maxId)
      feats = genEmbedding(model, graph)
      feats = features.toDF("node", "feature").persist($(storageLevel))

      val rdd_feature_merged = featureList
        .join(feats.toDF("node", "feature2"), "node")
        .rdd
        .map(row => (row.getLong(0),
          row.getString(1) + " " + row.getString(2).replace(',', ' ')))
      // Delimiter of features and smoothed features should both be a space instead of a comma

      featureList = features.sparkSession.createDataFrame(rdd_feature_merged).toDF("node", "feature")
    }
    featureList
  }

}
