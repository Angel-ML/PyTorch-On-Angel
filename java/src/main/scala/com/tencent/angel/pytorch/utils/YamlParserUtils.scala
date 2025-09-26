package com.tencent.angel.pytorch.utils

import com.tencent.angel.graph.utils.Delimiter
import groovy.json.JsonOutput.prettyPrint
import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.CustomClassLoaderConstructor

import java.io.{File, FileInputStream}
import java.util
import scala.beans.BeanProperty
import scala.collection.JavaConverters._
import scala.collection.mutable

object YamlParserUtils {

  def main(args: Array[String]): Unit = {
    parseGNN("./gnn.yaml", Constants.GRAPH_TYPE_HOMOGENEOUS)
  }

  def parseRecommendation(path: String): RecommendationParameter = {
    val input = new FileInputStream(new File(path))
    val yaml = new Yaml(new CustomClassLoaderConstructor(classOf[RecommendationParameter],
      Thread.currentThread().getContextClassLoader))
    val config: RecommendationParameter = yaml.load(input).asInstanceOf[RecommendationParameter]
    config.data.putIfAbsent(Constants.TRAIN_INPUT, "")
    config.data.putIfAbsent(Constants.VALIDATE_INPUT, "")
    config.data.putIfAbsent(Constants.TDW_COLUMN_INDEX, "0")
    config.trainer.putIfAbsent(Constants.BATCH_SIZE, "512")
    config.trainer.putIfAbsent(Constants.LEARNING_RATE, "0.01")
    config.trainer.putIfAbsent(Constants.TEST_RATIO, "0.1")
    config.trainer.putIfAbsent(Constants.OPTIMIZER, "adam")
    config.trainer.putIfAbsent(Constants.DECAY, "0.001")
    config.trainer.putIfAbsent(Constants.EPOCH, "10")
    config.trainer.putIfAbsent(Constants.ASYNC, "true")
    config.trainer.putIfAbsent(Constants.MODEL_SAVE_PATH, "")
    config.trainer.putIfAbsent(Constants.MODEL_LOAD_PATH, "")
    config.trainer.putIfAbsent(Constants.SAVE_TORCH_MODEL, "false")
    config.trainer.putIfAbsent(Constants.ROW_TYPE, "T_FLOAT_DENSE")
    config.trainer.putIfAbsent(Constants.EVAL_METRICS, "auc")
    config.trainer.putIfAbsent(Constants.SAVE_FEAT_IMPORTANCE, "false")
    config.trainer.putIfAbsent(Constants.SHUFFLE_INTERVAL, "1000000")
    config.trainer.putIfAbsent(Constants.CALC_VALIDATE_LOSS, "false")
    config.predictor.putIfAbsent(Constants.PREDICT_OUTPUT_PATH, "")
    config.predictor.putIfAbsent(Constants.MODEL_LOAD_PATH, "")
    config.predictor.putIfAbsent(Constants.SAVE_PREDICTED_VECTOR, "false")
    config.model.putIfAbsent(Constants.DENSE_DIM, "0")
    println(config.toString)
    config
  }

  def parseGNN(path: String, graphType: String): GNNParameter = {
    val input = new FileInputStream(new File(path))
    val yaml = new Yaml(new CustomClassLoaderConstructor(classOf[GNNParameter],
      Thread.currentThread().getContextClassLoader))
    val config: GNNParameter = yaml.load(input).asInstanceOf[GNNParameter]
    config.model.putIfAbsent(Constants.SECOND_ORDER, "true")
    if (Constants.GRAPH_TYPE_HOMOGENEOUS.equals(graphType)) {
      val node = config.graph.nodes.get(0)
      node.feature.putIfAbsent(Constants.PATH, "")
      node.feature.putIfAbsent(Constants.FORMAT, "sparse")
      node.feature.putIfAbsent(Constants.DELIMITER, "tab")

      config.graph.edges.base_config.putIfAbsent(Constants.WEIGHTED, "false")
      config.graph.edges.base_config.putIfAbsent(Constants.EDGE_TIMESTAMP, "false")
      config.graph.edges.base_config.putIfAbsent(Constants.DELIMITER, "space")
      val edge = config.graph.edges.link_config.get(0)
      edge.putIfAbsent(Constants.PATH, "")

      parseGNNTrainerBase(config)
      config.trainer.putIfAbsent(Constants.SAMPLE_NUM, "5")
      config.trainer.putIfAbsent(Constants.HOPS, "2")
      config.trainer.putIfAbsent(Constants.USE_WEIGHTED_AGGREGATE, "false")
      config.trainer.putIfAbsent(Constants.SAVE_CONCAT_FEATS, "true")
      config.trainer.putIfAbsent(Constants.SHUFFLE_SAMPLES, "false")
      config.trainer.putIfAbsent(Constants.AUGMENTER, "NodeDropping")
      config.trainer.putIfAbsent(Constants.AUGMENT_RATIO, "0.3")

      parseGNNPredictorBase(config)

      config.model.putIfAbsent(Constants.INPUT_DIM, "-1")
      config.model.putIfAbsent(Constants.INPUT_EMBEDDING_DIM, "-1")
      config.model.putIfAbsent(Constants.INPUT_FIELD_NUM, "-1")
      config.model.putIfAbsent(Constants.AGGREGATION_TYPE, "mean")
      config.model.putIfAbsent(Constants.INPUT_SPLIT_IDX, "0")
      config.model.putIfAbsent(Constants.NEG_SAMPLING, "true")
      config.model.putIfAbsent(Constants.INPUT_EDGE_EMBEDDING_DIM, "-1")
      config.model.putIfAbsent(Constants.INPUT_EDGE_FIELD_NUM, "-1")
      config.model.putIfAbsent(Constants.EDGE_INPUT_SPLIT_IDX, "0")
    } else if (Constants.GRAPH_TYPE_BIPARTITE.equals(graphType)) {
      parseGNNNodeAndEdgesBase(config)

      parseGNNTrainerBase(config)
      config.trainer.putIfAbsent(Constants.USER_SAMPLE_NUM, "5")
      config.trainer.putIfAbsent(Constants.ITEM_SAMPLE_NUM, "5")

      parseGNNPredictorBase(config)

      config.model.putIfAbsent(Constants.INPUT_USER_DIM, "-1")
      config.model.putIfAbsent(Constants.INPUT_ITEM_DIM, "-1")
      config.model.putIfAbsent(Constants.INPUT_USER_EMBEDDING_DIM, "-1")
      config.model.putIfAbsent(Constants.INPUT_USER_FIELD_NUM, "-1")
      config.model.putIfAbsent(Constants.INPUT_ITEM_EMBEDDING_DIM, "-1")
      config.model.putIfAbsent(Constants.INPUT_ITEM_FIELD_NUM, "-1")
      config.model.putIfAbsent(Constants.USER_INPUT_SPLIT_IDX, "0")
      config.model.putIfAbsent(Constants.ITEM_INPUT_SPLIT_IDX, "0")
      config.model.putIfAbsent(Constants.INPUT_EDGE_DIM, "-1")
      config.model.putIfAbsent(Constants.INPUT_EDGE_EMBEDDING_DIM, "-1")
      config.model.putIfAbsent(Constants.INPUT_EDGE_FIELD_NUM, "-1")
      config.model.putIfAbsent(Constants.EDGE_INPUT_SPLIT_IDX, "0")
      config.model.putIfAbsent(Constants.ENCODE, "dense")
      config.model.putIfAbsent(Constants.M, "-1")
      config.model.putIfAbsent(Constants.TASK_TYPE, "classification")
      config.model.putIfAbsent(Constants.ITEM_TYPES, "-1")
      config.model.putIfAbsent(Constants.EDGE_TYPES, "-1")
      config.model.putIfAbsent(Constants.NODE_TYPES, "-1")
      config.model.putIfAbsent(Constants.FFN_HIDDEN_SIZE, "-1")
      config.model.putIfAbsent(Constants.LAYERS, "-1")
      config.model.putIfAbsent(Constants.NEG_SAMPLING, "true")
      config.model.putIfAbsent(Constants.IS_OPTIMIZE, "true")
      config.model.putIfAbsent(Constants.CONV, "bisage")

    } else if (Constants.GRAPH_TYPE_HETEROGENEOUS.equals(graphType)) {
      parseGNNNodeAndEdgesBase(config)
      parseGNNTrainerBase(config)
      config.trainer.putIfAbsent(Constants.NODES_SAMPLE_NUM, "")
      config.trainer.putIfAbsent(Constants.SUB_GRAPH_SAMPLE_NUM, "100")
      config.trainer.putIfAbsent(Constants.FILTER_SAME_NODE, "true")
      config.trainer.putIfAbsent(Constants.KEY_NODE, "u")
      config.model.putIfAbsent(Constants.EDGE_TYPES, "-1")
      config.trainer.putIfAbsent(Constants.TEST_EDGE_PATH, "")
      config.model.putIfAbsent(Constants.EMBEDDING_DIM, "200")
      config.trainer.putIfAbsent(Constants.PARTITION_OPT, "false")
      config.trainer.putIfAbsent(Constants.LOCAL_NEGATIVE_SAMPLE, "false")
      parseGNNPredictorBase(config)
    }
    println(config.toString)
    config
  }

  def parseGNNTrainerBase(config: GNNParameter): Unit = {
    config.trainer.putIfAbsent(Constants.BATCH_SIZE, "128")
    config.trainer.putIfAbsent(Constants.LEARNING_RATE, "0.01")
    config.trainer.putIfAbsent(Constants.OPTIMIZER, "adam")
    config.trainer.putIfAbsent(Constants.EPOCH, "10")
    config.trainer.putIfAbsent(Constants.TEST_RATIO, "0.5")
    config.trainer.putIfAbsent(Constants.BATCH_INIT_BUM, "5")
    config.trainer.putIfAbsent(Constants.PERIODS, "100")
    config.trainer.putIfAbsent(Constants.CHECKPOINT_INTERVAL, "100")
    config.trainer.putIfAbsent(Constants.SAVE_MODEL_INTERVAL, "100")
    config.trainer.putIfAbsent(Constants.DECAY, "0.0")
    config.trainer.putIfAbsent(Constants.EVAL_METRICS, "acc")
    config.trainer.putIfAbsent(Constants.MULTI_HOT_FIELD, "false")
    config.trainer.putIfAbsent(Constants.THREADS_NUM, "5")
    config.trainer.putIfAbsent(Constants.PARALLEL_TRAIN, "false")
    config.trainer.putIfAbsent(Constants.VALIDATE_PERIODS, "5")
    config.trainer.putIfAbsent(Constants.USE_SHARED_SAMPLES, "false")
    config.trainer.putIfAbsent(Constants.SAVE_CHECKPOINT, "false")
    config.trainer.putIfAbsent(Constants.SAVE_TORCH_MODEL, "false")
    config.trainer.putIfAbsent(Constants.SAMPLE_METHOD, "random")
    config.trainer.putIfAbsent(Constants.EVALUATION_RATIO, "1.0")
    config.trainer.putIfAbsent(Constants.MODEL_SAVE_PATH, "")
    config.trainer.putIfAbsent(Constants.FEAT_EMBEDDING_LOAD_PATH, "")
    config.trainer.putIfAbsent(Constants.LABELS_NUM, "1")
    config.trainer.putIfAbsent(Constants.TRAIN_SAMPLE_RATIO, "0.5")
    config.trainer.putIfAbsent(Constants.INIT_METHOD, "xavierUniform")
    config.trainer.putIfAbsent(Constants.MEAN, "0")
    config.trainer.putIfAbsent(Constants.STD, "1")
    config.trainer.putIfAbsent(Constants.WINDOW_SIZE, "10")
    config.trainer.putIfAbsent(Constants.NEGATIVE_SAMPLE_NUM, "10")
    config.trainer.putIfAbsent(Constants.LOG_STEP, "1000")
    config.trainer.putIfAbsent(Constants.NEGATIVE_SAMPLE_BY_NODE_TYPE, "false")
    config.trainer.putIfAbsent(Constants.EVALUATE_BY_EDGE_TYPE, "false")
    config.trainer.putIfAbsent(Constants.OUTPUT_EMBEDDING_BY_NODE_TYPE, "false")
    config.trainer.putIfAbsent(Constants.OUTPUT_EMBEDDING_BY_EDGE_TYPE, "false")
    config.trainer.putIfAbsent(Constants.INDEPENDENT_FIELD, "false")
  }

  def parseGNNPredictorBase(config: GNNParameter): Unit = {
    config.predictor.putIfAbsent(Constants.CHECK_OUTPUT, "true")
    config.predictor.putIfAbsent(Constants.BATCH_SIZE_MULTIPLIER, "10")
    config.predictor.putIfAbsent(Constants.NODE_EMBEDDING_OUTPUT_PATH, "")
    config.predictor.putIfAbsent(Constants.PREDICT_OUTPUT_PATH, "")
  }

  def parseGNNNodeAndEdgesBase(config: GNNParameter): Unit = {
    val nodeNames = new mutable.HashSet[String]()
    for (node <- config.graph.nodes.asScala) {
      require(node.name != null, "node name in config file must not null!")
      node.feature.putIfAbsent(Constants.PATH, "")
      node.feature.putIfAbsent(Constants.FORMAT, "sparse")
      node.feature.putIfAbsent(Constants.DELIMITER, "tab")
      nodeNames.add(node.name)
    }
    config.graph.edges.base_config.putIfAbsent(Constants.WEIGHTED, "false")
    config.graph.edges.base_config.putIfAbsent(Constants.EDGE_TIMESTAMP, "false")
    config.graph.edges.base_config.putIfAbsent(Constants.DELIMITER, "space")
    for(edge <- config.graph.edges.link_config.asScala) {
      if (edge.containsKey(Constants.SRC) || edge.containsKey(Constants.DST)) {
        require(edge.containsKey(Constants.SRC) && nodeNames.contains(edge.get(Constants.SRC).toString),
          "src and dst name must be exist and should same as node name.")
        require(edge.containsKey(Constants.DST) && nodeNames.contains(edge.get(Constants.DST).toString),
          "src and dst name must be exist and should same as node name.")
      }
      edge.putIfAbsent(Constants.PATH, "")
    }
  }

  def parseNodePaths(nodes: util.ArrayList[Node]): mutable.HashMap[String, (String, String)] = {
    val nodePathsMap = mutable.HashMap[String, (String, String)]()
    for (node <- nodes.asScala) {
      nodePathsMap.put(node.name, (FileUtils.parseDatePartitionName(node.feature.get(Constants.PATH).toString),
        Delimiter.parse(node.feature.get(Constants.DELIMITER).toString)))
    }
    nodePathsMap
  }

  def parseQueryPaths(nodes: util.ArrayList[Node]): mutable.HashMap[String, String] = {
    val queryPathsMap = mutable.HashMap[String, String]()
    for (node <- nodes.asScala) {
      if (node.query != null && node.query.get(Constants.PATH) != null) {
        queryPathsMap.put(node.name, FileUtils.parseDatePartitionName(node.query.get(Constants.PATH).toString))
      }
    }
    queryPathsMap
  }

  def parseEdgePaths(edges: util.ArrayList[util.HashMap[String, Object]]): mutable.HashMap[String, String] = {
    val edgePathsMap = mutable.HashMap[String, String]()
    for (edge <- edges.asScala) {
      edgePathsMap.put(edge.get(Constants.SRC) + "-" + edge.get(Constants.DST),
        FileUtils.parseDatePartitionName(edge.get(Constants.PATH).toString))
    }
    edgePathsMap
  }

  def parseSrcNames(edges: util.ArrayList[util.HashMap[String, Object]]): mutable.HashSet[String] = {
    val srcNames = mutable.HashSet[String]()
    for (edge <- edges.asScala) {
      srcNames.add(edge.get(Constants.SRC).toString)
    }
    srcNames
  }

  def getFeatureAndLabelPaths(config: GNNParameter): (String, String, String, String, String, String) = {
    var userFeatureInput = ""
    var itemFeatureInput = ""
    var labelPath = ""
    var testLabelPath = ""
    var userFeatureSep = ""
    var itemFeatureSep = ""
    val nodes = config.graph.nodes
    for (node <- nodes.asScala) {
      if(node.name.equals(config.graph.edges.link_config.get(0).get(Constants.SRC).toString)) {
        userFeatureInput = FileUtils.parseDatePartitionName(node.feature.get(Constants.PATH).toString)
        userFeatureSep = node.feature.get(Constants.DELIMITER).toString
        labelPath = FileUtils.parseDatePartitionName(node.label.get(Constants.PATH).toString)
        testLabelPath = FileUtils.parseDatePartitionName(node.label.get(Constants.VALIDATE_PATH).toString)
      } else if (node.name.equals(config.graph.edges.link_config.get(0).get(Constants.DST).toString)) {
        itemFeatureInput = FileUtils.parseDatePartitionName(node.feature.get(Constants.PATH).toString)
        itemFeatureSep = node.feature.get(Constants.DELIMITER).toString
      }
    }
    (userFeatureInput, itemFeatureInput, labelPath, testLabelPath, userFeatureSep, itemFeatureSep)
  }

  def getFeatureAndQueryPaths(config: GNNParameter): (String, String, String, String, String, String) = {
    var userFeatureInput = ""
    var itemFeatureInput = ""
    var userQueryPath = ""
    var itemQueryPath = ""
    var userFeatureSep = ""
    var itemFeatureSep = ""
    val nodes = config.graph.nodes
    for (node <- nodes.asScala) {
      if(node.name.equals(config.graph.edges.link_config.get(0).get(Constants.SRC).toString)) {
        userFeatureInput = FileUtils.parseDatePartitionName(node.feature.get(Constants.PATH).toString)
        userFeatureSep = node.feature.get(Constants.DELIMITER).toString
        if (node.query != null) {
          userQueryPath = FileUtils.parseDatePartitionName(node.query.get(Constants.PATH).toString)
        }
      } else if (node.name.equals(config.graph.edges.link_config.get(0).get(Constants.DST).toString)) {
        itemFeatureInput = FileUtils.parseDatePartitionName(node.feature.get(Constants.PATH).toString)
        itemFeatureSep = node.feature.get(Constants.DELIMITER).toString
        if (node.query != null) {
          itemQueryPath = FileUtils.parseDatePartitionName(node.query.get(Constants.PATH).toString)
        }
      }
    }
    (userFeatureInput, itemFeatureInput, userQueryPath, itemQueryPath, userFeatureSep, itemFeatureSep)
  }

  class RecommendationParameter {
    @BeanProperty var data: util.HashMap[String, Object] = _
    @BeanProperty var model: util.HashMap[String, Object] = _
    @BeanProperty var trainer: util.HashMap[String, Object] = _
    @BeanProperty var predictor: util.HashMap[String, Object] = _
    override def toString: String = prettyPrint(groovy.json.JsonOutput.toJson(this))
  }

  class GNNParameter {
    @BeanProperty var graph: Graph = _
    @BeanProperty var model: util.HashMap[String, Object] = _
    @BeanProperty var trainer: util.HashMap[String, Object] = _
    @BeanProperty var predictor: util.HashMap[String, Object] = _
    override def toString: String = prettyPrint(groovy.json.JsonOutput.toJson(this))
  }

  class Graph {
    @BeanProperty var name: String = _
    @BeanProperty var nodes: util.ArrayList[Node] = _
    @BeanProperty var edges: Edges = _
  }

  class Node {
    @BeanProperty var name: String = _
    @BeanProperty var feature: util.HashMap[String, Object] = _
    @BeanProperty var label: util.HashMap[String, Object] = _
    @BeanProperty var query: util.HashMap[String, Object] = _
  }

  class Edges {
    @BeanProperty var base_config: util.HashMap[String, Object] = _
    @BeanProperty var label: util.HashMap[String, Object] = _
    @BeanProperty var link_config: util.ArrayList[util.HashMap[String, Object]] = _
  }
}
