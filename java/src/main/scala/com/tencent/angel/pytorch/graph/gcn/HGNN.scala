package com.tencent.angel.pytorch.graph.gcn

import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.io.{DataLoaderUtils, IOFunctions}
import com.tencent.angel.pytorch.params._
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.context.PSContext
import org.apache.spark.SparkContext
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import scala.collection.mutable.HashMap


class HGNN extends GNN with HasMetaPaths with HasFeatureDims with HasNodeNumSamples with HasTotalNodes
  with HasFieldNums with HasEmbedDims with HasTestRatio with HasFieldMultiHot with HasFilterSameNode
  with HasFeatureSplitIdxs {

  def initialize(edgeDFs: Map[String, DataFrame], featureDFs: Map[String, DataFrame]
                ): (HGNNPSModel, Dataset[_]) =
    initialize(edgeDFs, featureDFs, None, None)

  def initialize(edgeDFs: Map[String, DataFrame], featureDFs: Map[String, DataFrame],
                 queryDFs: Option[Map[String, DataFrame]]): (HGNNPSModel, Dataset[_]) =
    initialize(edgeDFs, featureDFs, queryDFs, None)

  def initialize(edgeDFs: Map[String, DataFrame], featureDFs: Map[String, DataFrame],
                 labelDF: Option[Map[String, DataFrame]], testLabelDF: Option[Map[String, DataFrame]]
                ): (HGNNPSModel, Dataset[_]) = {

    var startTs = System.currentTimeMillis()

    edgeDFs.foreach(_._2.persist($(storageLevel)))

    val nodesMap = new HashMap[String, (Long, Long)]()

    //statics minId and maxId for each edge
    staticIds(edgeDFs, nodesMap)

    // create weights, graph on servers
    TorchModel.setPath($(torchModelPath))
    val torch = TorchModel.get()
    val weightsSize = torch.getParametersTotalSize
    println(s"weight total size=${weightsSize}")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    setMetaPaths(torch.getMetaPaths)
    setTotalNodes(torch.getTotalNodes)
    setFieldNums(torch.getFieldNums)
    setEmbedDims(torch.getFeatEmbedDims)
    setFeatureDims(torch.getFeatureDims)
    if ($(dataFormat) == FeatureFormat.DENSE_HIGH_SPARSE) setFeatureSplitIdxs(torch.getFeatureSplitIdxs)

    val featureIds = if ($(fieldNums).length > 0) Option(getFeatureIndexs(featureDFs)) else None
    if (featureIds.nonEmpty) featureIds.get.foreach(_._2.persist($(storageLevel)))

    // init ps model
    val model = HGNNPSModel(nodesMap, featureIds, weightsSize, getOptimizer, getPartitionIndexs(edgeDFs),
      getEmbedDims, getFeatureDims, $(psPartitionNum), $(useBalancePartition), labelDF, testLabelDF,
      getFeatureSplitIdxs)

    model.setWeights(torch.getParameters)
    TorchModel.put(torch)

    if (labelDF.nonEmpty) initQueries(model, labelDF.get, nodesMap)
    if (testLabelDF.nonEmpty) initQueries(model, testLabelDF.get, nodesMap)

    if ($(featEmbedPath).length > 0) {
      initExtraEmbeddings(model, $(featEmbedPath), nodesMap.keys.toArray)
    } else {
      if (torch.getFieldNums.size > 0)
        model.asInstanceOf[HGNNPSModel].initEmbeddings(featureIds.get, $(batchSize), getOptimizer.getNumSlots(), getInitMethod)
    }

    // init neighbors
    val graphs = makeGraphs(edgeDFs, model, labelDF, testLabelDF)

    // init features
    initFeatures(model, featureDFs, nodesMap)

    if (torch.getFeatEmbedDims.size > 0) setFeatureDims(getEmbedDims.map(p => p._1 + ":" + p._2).mkString(","))

    println(s"initialize cost ${(System.currentTimeMillis() - startTs) / 1000}s")

    startTs = System.currentTimeMillis()
    if ($(saveCheckpoint))
      model.checkpointMatrices(0)
    println(s"Write checkpoint use time=${System.currentTimeMillis() - startTs}ms")

    (model, graphs)
  }

  def getMinMaxId(edges: DataFrame, nodeType: String): (Long, Long, Long) =
  //nodeType: src or dst
    edges.select(nodeType).rdd
      .map(row => (row.getLong(0), row.getLong(0)))   // count only one type
      .mapPartitions(DataLoaderUtils.summarizeApplyOp)
      .reduce(DataLoaderUtils.summarizeReduceOp)

  def staticIds(edgeDFs: Map[String, DataFrame], nodesMap: HashMap[String, (Long, Long)]): Unit = {
    edgeDFs.foreach{ case (edgeName, df) =>
      val nodeTypes = Array("src", "dst")
      val nodes = edgeName.split("-")

      nodeTypes.zip(nodes).foreach{ case (nType, node) =>
        var (minId, maxId, numEdges) = getMinMaxId(df, nType)
        if (nodesMap.contains(node)) {
          val (minIdOld, maxIdOld) = nodesMap.get(node).get
          minId = if (minId < minIdOld) minId else minIdOld
          maxId = if (maxId > maxIdOld) maxId else maxIdOld
        }
        nodesMap.put(node, (minId, maxId))
        println(s"edge=$edgeName node=$node minId=$minId maxId=$maxId numEdges=$numEdges")
      }
    }
  }

  def makeGraphs(edgeDFs: Map[String, DataFrame], model: HGNNPSModel,
                 labelDF: Option[Map[String, DataFrame]],
                 testLabelDF: Option[Map[String, DataFrame]]): Dataset[_] = ???

  def initQueries(model: HGNNPSModel, queries: Map[String, DataFrame], nodeIds: HashMap[String, (Long, Long)]): Unit = {
    queries.map{ case (name, query) =>
      println(s"query:  ${name}: ${query.count()}")
      val (minId, maxId) = nodeIds.get(name).get
      query.rdd.map(row => row.getLong(0)).map(x => (x, 0.0f))
        .filter(f => f._1 >= minId && f._1 <= maxId)
        .mapPartitionsWithIndex((index, it) =>
          Iterator.single(NodeLabelPartition.apply(index, it, maxId)))
        .map(_.init(model, name)).count()
    }
  }

  def initFeatures(model: HGNNPSModel, features: Map[String, Dataset[Row]], nodeIds: HashMap[String, (Long, Long)]): Unit = {
    val dimsMap = getFeatureDims
    val featureSplitIdxs = getFeatureSplitIdxs

    features.map{ case (name, feat) =>
      val (minId, maxId) = nodeIds.getOrElse(name, (-1L, -1L))
      feat.select("node", "feature").rdd.filter(row => row.length > 0)
        .filter(row => row.get(0) != null)
        .map(row => (row.getLong(0), row.getString(1)))
        .filter(f => f._1 >= minId && f._1 <= maxId)
        .map(f => (f._1, SampleParser.parseFeature(f._2, dimsMap.getOrElse(name, -1L).toInt, $(dataFormat), featureSplitIdxs.getOrElse(name, 0))))
        .repartition($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator.single(NodeFeaturePartition.apply(index, it)))
        .map(_.init(model, name, $(numBatchInit))).count()
    }
  }

  def initExtraEmbeddings(model: HGNNPSModel, featEmbedPath: String, nodeNames: Array[String]): Unit = {
    val conf = SparkHadoopUtil.get.newConfiguration(SparkContext.getOrCreate().getConf)
    val keyValueSep = IOFunctions.parseSep(conf.get("angel.embedding.keyvalue.sep", "colon"))
    val featSep = IOFunctions.parseSep(conf.get("angel.embedding.feature.sep", "space"))

    nodeNames.foreach{ name =>
      val path = featEmbedPath + "/" + name + "Embedding"
      IOFunctions.loadFeature(path, sep = keyValueSep).rdd
        .map(r => (r.getLong(0), SampleParser.parseEmbedding(r.getString(1), featSep)))
        .repartition($(partitionNum))
        .mapPartitionsWithIndex((index, it) =>
          Iterator.single(NodeFeaturePartition.apply(index, it)))
        .map(_.init(model, name, $(numBatchInit), getOptimizer.getNumSlots())).count()

    }
  }

  def fit(model: HGNNPSModel, graphs: Dataset[_], checkPointPath: String): Unit = ???

  def getPartitionIndexs(edgeDFs: Map[String, DataFrame]): Map[String, RDD[Long]] = {
    val nodes = new HashMap[String, RDD[Long]]()
    edgeDFs.foreach{ case (name, df) =>
      name.split("-").zipWithIndex.foreach{ case(key, idx) =>
        val node = df.select("src", "dst").rdd.map(row => row.getLong(idx))
        if (nodes.contains(key)) {
          val rdd = nodes.get(key).get
          nodes.put(key, rdd.union(node))
        }
        nodes.put(key, node)
      }
    }
    nodes.toMap
  }

  def getFeatureIndexs(featureDFs: Map[String, DataFrame]): Map[String, RDD[Long]] = {
    val dimsMap = getFeatureDims

    featureDFs.map{ case (name, df) =>
      val fid = getFeatureIds(df, dimsMap.getOrElse(name, -1L))
      (name, fid)
    }
  }

  def genEmbedding(model: HGNNPSModel, graph: Dataset[_], name: String): DataFrame = {
    val dimsMap = getFeatureDims
    val fieldNumsMap = getFieldNums
    val samplesMap = getNodeNumSamples
    val mPaths = $(metaPaths).toString.split(",")
    val embedDims = getEmbedDims
    val featureSplitIdxs = getFeatureSplitIdxs

    val ret = graph.rdd.flatMap(_.asInstanceOf[HGNNPartition]
      .genEmbedding($(batchSize) * $(batchSizeMultiple), model, samplesMap,
        dimsMap.map(p => (p._1, p._2.toInt)), graph.rdd.getNumPartitions, name,
        fieldNumsMap, $(fieldMultiHot), mPaths, $(totalNodes), $(filterSameNode), embedDims, featureSplitIdxs))
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("embedding", StringType, nullable = false)
    ))

    graph.sparkSession.createDataFrame(ret, schema)
  }

  /**
   * Save feature embedding of sparse feature
   * @param model
   * @param savePath
   * @param curEpoch
   */
  def saveFeatEmbeds(model: HGNNPSModel, savePath: String, curEpoch: Int = -1): Unit = {
    if (getFieldNums.size > 0) {
      val path = if (curEpoch < 0) savePath + "/featEmbed" else savePath + s"/featEmbed_$curEpoch"
      model.saveFeatEmbed(path)
    }
  }
}