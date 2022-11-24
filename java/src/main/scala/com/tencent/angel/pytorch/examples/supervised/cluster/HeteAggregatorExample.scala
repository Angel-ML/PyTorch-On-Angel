package com.tencent.angel.pytorch.examples.supervised.cluster

import com.tencent.angel.pytorch.graph.gcn.HAggregator
import com.tencent.angel.pytorch.io.IOFunctions
import com.tencent.angel.pytorch.utils.{FileUtils, PartitionUtils}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.graph.utils.GraphIO
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel.MEMORY_ONLY
import org.apache.spark.{SparkConf, SparkContext}

object HeteAggregatorExample {
  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val edgeInputs = params.getOrElse("edgePaths", "") //u-i:path,i-v:path, ...
    val featureInputs = params.getOrElse("featurePaths", "") //u:path,i:path,v:path, ...
    val featureDims = params.getOrElse("featureDims", "") //u:u_dim,v:v_dim,i:i_dim
    val metapathsInput = params.getOrElse("metapaths", "") // u-i,i-v,v-i,i-u
    val embeddingOutputPaths = params.getOrElse("embeddingOutputPaths", "")//u:u_path,i:i_path,v:v_path, ...
    val featureEmbedInputPath = params.getOrElse("featureEmbedInputPath", "")
    val fieldMultiHot = params.getOrElse("fieldMultiHot", "false").toBoolean
    val batchSize = params.getOrElse("batchSize", "100").toInt
    var torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val numSamples = params.getOrElse("samples", "5").toInt
    var psNumPartition = params.getOrElse("psNumPartition", "10").toInt
    var numPartitions = params.getOrElse("numPartitions", "5").toInt
    val psNumPartitionFactor = params.getOrElse("psNumPartitionFactor", "2").toInt
    val numPartitionsFactor = params.getOrElse("numPartitionsFactor", "3").toInt
    val useBalancePartition = params.getOrElse("useBalancePartition", "false").toBoolean
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val format = params.getOrElse("format", "sparse")
    val actionType = params.getOrElse("actionType", "train")
    val saveCheckpoint = params.getOrElse("saveCheckpoint", "false").toBoolean
    val batchSizeMultiple = params.getOrElse("batchSizeMultiple", "10").toInt
    val makeDenseOutput = params.getOrElse("makeDenseOutput", "false").toBoolean
    val isWeighted = params.getOrElse("isWeighted", "false").toBoolean
    val useWeightedAggregate = if (isWeighted) params.getOrElse("useWeightedAggregate", "false").toBoolean else false
    val aggregator_in_scala = if (useWeightedAggregate) true else params.getOrElse("aggregator_in_scala", "false").toBoolean
    val sep = params.getOrElse("sep", "space") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }
    val featureSep = params.getOrElse("featureSep", "tab") match {
      case "space" => " "
      case "comma" => ","
      case "tab" => "\t"
    }

    val conf = new SparkConf()
    conf.set("spark.executor.extraLibraryPath", "./torch/torch-lib")
    conf.set("spark.executorEnv.OMP_NUM_THREADS", "2")
    conf.set("spark.executorEnv.MKL_NUM_THREADS", "2")

    val sc = new SparkContext(conf)

    numPartitions = PartitionUtils.getDataPartitionNum(numPartitions, conf, numPartitionsFactor)
    psNumPartition = PartitionUtils.getPsPartitionNum(psNumPartition, conf, psNumPartitionFactor)
    println(s"numPartition: $numPartitions, numPsPartition: $psNumPartition")

    torchModelPath = FileUtils.getPtName("./")
    println("torchModelPath is: " + torchModelPath)

    val gcn = new HAggregator()
    gcn.setTorchModelPath(torchModelPath)
    gcn.setUseBalancePartition(false)
    gcn.setBatchSize(batchSize)
    gcn.setPSPartitionNum(psNumPartition)
    gcn.setPartitionNum(numPartitions)
    gcn.setUseBalancePartition(useBalancePartition)
    gcn.setStorageLevel(storageLevel)
    gcn.setDataFormat(format)
    gcn.setNumSamples(numSamples)
    gcn.setSaveCheckpoint(saveCheckpoint)
    gcn.setBatchSizeMultiple(batchSizeMultiple)
    gcn.setFeatEmbedPath(featureEmbedInputPath)
    gcn.setFieldMultiHot(fieldMultiHot)
    gcn.setHasWeighted(isWeighted)
    gcn.setHasUseWeightedAggregate(useWeightedAggregate)

    val edgesMap = edgeInputs.split(",").map { pair =>
      val kv = pair.split(":", 2) // name:path
      (kv(0), GraphIO.load(kv(1), isWeighted = isWeighted, sep = sep))
    }.toMap
    val featuresMap = featureInputs.split(",").map { pair =>
      val kv = pair.split(":", 2) // name:path
      (kv(0), IOFunctions.loadFeature(kv(1), sep = featureSep))
    }.toMap

    // recode the node id to a continuous space
    val (node2idMap, nodeType2id) = gcn.recodeNodes(featuresMap)
    val flatten_features = gcn.flattenFeatures(featuresMap, node2idMap, nodeType2id).persist(MEMORY_ONLY)  // node, type, feature
    val (edges, edgeType2id) = gcn.flattenEdges(edgesMap, node2idMap)
    val metapaths = metapathsInput.split(",").map(r => edgeType2id(r.split("-").reverse.mkString("-")))

    println("metapaths")
    metapaths.foreach(println)
    val dims = featureDims.split(",").map{ pair =>
      val kv = pair.split(":", 2) // name:dim
      (nodeType2id(kv(0)), kv(1).toInt)
    }
    val (temp_features, dim) = gcn.concatFeatures(flatten_features, dims)
    var features = temp_features
    gcn.setFeatureDim(dim)
    println("featureDim", dim)

    val (model, graph) = gcn.initialize(edges, features, None)

    assert(embeddingOutputPaths.nonEmpty)
    val (minId, maxId, _) = gcn.getMinMaxId(edges)

    var featureList = features
    val spark = SparkSession.builder.getOrCreate
    import spark.implicits._

    for (hop <- metapaths.indices) {
      println("Aggregate with " + hop + " hop")
      println("Before aggregating")
      gcn.initFeatures(model, features, minId, maxId)
      var new_features = gcn.genEmbedding(model, graph, metapaths(hop), aggregator_in_scala)
      new_features = new_features.toDF("node", "feature").persist(MEMORY_ONLY)

      println("After aggregating")
      features = features.select("node", "feature").rdd.map(row => (row.getLong(0), row.getString(1)))
        .subtractByKey(new_features.select("node", "feature").rdd.map(row => (row.getLong(0), row.getString(1))), numPartitions)
        .union(new_features.select("node", "feature").rdd.map(row => (row.getLong(0), row.getString(1))))
        .toDF("node", "feature")
      println("subtractByKey done.")

      val rdd_feature_merged = featureList
        .join(features.toDF("node", "feature2"), "node")
        .rdd
        .map(row => (row.getLong(0),
          row.getString(1) + " " + row.getString(2).replace(',', ' ')))
      // Delimiter of features and smoothed features should both be a space instead of a comma
      featureList = spark.createDataFrame(rdd_feature_merged).toDF("node", "feature")
    }

    embeddingOutputPaths.split(",").foreach{ pair =>
      val kv = pair.split(":", 2)// name:path
      val embedding = gcn.getNodeTypeFeatures(featureList, node2idMap, kv(0))
      println("Write node type " + kv(0))
      GraphIO.save(embedding, kv(1), seq = "\t")
      }

    PSContext.stop()
    sc.stop()
  }
}
