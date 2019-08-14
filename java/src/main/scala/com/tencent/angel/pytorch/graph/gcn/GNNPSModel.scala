package com.tencent.angel.pytorch.graph.gcn

import java.util.{ArrayList => JArrayList}

import com.tencent.angel.graph.client.getnodefeats2.{GetNodeFeats, GetNodeFeatsParam, GetNodeFeatsResult}
import com.tencent.angel.graph.client.initNeighbor4.{InitNeighbor => InitNeighbor4, InitNeighborParam => InitNeighborParam4}
import com.tencent.angel.graph.client.initNeighbor5.{InitNeighbor => InitNeighbor5, InitNeighborParam => InitNeighborParam5}
import com.tencent.angel.graph.client.initnodefeats3.{InitNodeFeats => InitNodeFeats3, InitNodeFeatsParam => InitNodeFeatsParam3}
import com.tencent.angel.graph.client.initnodefeats4.{InitNodeFeats => InitNodeFeats4, InitNodeFeatsParam => InitNodeFeatsParam4}
import com.tencent.angel.graph.client.sampleFeats.{SampleNodeFeats, SampleNodeFeatsParam, SampleNodeFeatsResult}
import com.tencent.angel.graph.client.sampleneighbor3.{SampleNeighbor, SampleNeighborParam, SampleNeighborResult}
import com.tencent.angel.graph.client.summary.{NnzEdge, NnzFeature, NnzNeighbor, NnzNode}
import com.tencent.angel.graph.data.Node
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.vector.{IntFloatVector, LongFloatVector}
import com.tencent.angel.ml.matrix.psf.aggr.enhance.ScalarAggrResult
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.spark.ml.psf.gcn.{GetLabels, GetLabelsResult}
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.spark.models.impl.{PSMatrixImpl, PSVectorImpl}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import com.tencent.angel.spark.util.VectorUtils
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import org.apache.spark.rdd.RDD

private[gcn]
class GNNPSModel(graph: PSMatrix,
                 weights: PSVector,
                 labels: PSVector) extends Serializable {

  val dim: Long = labels.dimension

  // the default pull method will return keys even those not exists on servers
  def readLabels(keys: Array[Long]): LongFloatVector =
    labels.pull(keys.clone()).asInstanceOf[LongFloatVector]

  // this method will not return keys that do not exist on servers
  def readLabels2(keys: Array[Long]): LongFloatVector = {
    val func = new GetLabels(labels.poolId, keys.clone())
    labels.psfGet(func).asInstanceOf[GetLabelsResult].getVector
  }

  def setLabels(value: LongFloatVector): Unit =
    labels.update(value)

  def nnzLabels(): Long =
    VectorUtils.size(labels)

  def readWeights(): Array[Float] =
    weights.pull().asInstanceOf[IntFloatVector].getStorage.getValues

  def setWeights(values: Array[Float]): Unit = {
    val update = VFactory.denseFloatVector(values)
    weights.update(update)
  }

  def step(grads: Array[Float], optim: AsyncOptim): Unit = {
    val update = VFactory.denseFloatVector(grads)
    optim.asycUpdate(weights, 1, update).get()
  }

  /* summary functions */
  def nnzNodes(): Long = {
    val func = new NnzNode(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def nnzNeighbors(): Long = {
    val func = new NnzNeighbor(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def nnzFeatures(): Long = {
    val func = new NnzFeature(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def nnzEdge(): Long = {
    val func = new NnzEdge(graph.id, 0)
    graph.psfGet(func).asInstanceOf[ScalarAggrResult].getResult.toLong
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long]): Unit = {
    println(s"full batch init neighbors")
    val param = new InitNeighborParam4(graph.id, keys, indptr, neighbors)
    val func = new InitNeighbor4(param)
    graph.psfUpdate(func).get()
  }


  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    numBatch: Int): Unit = {
    println(s"mini batch init neighbors")
    val step = keys.length / numBatch
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initNeighbors(keys, indptr, neighbors, start, end)
      start += step
    }
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    start: Int,
                    end: Int): Unit = {
    val param = new InitNeighborParam5(graph.id, keys, indptr, neighbors, start, end)
    val func = new InitNeighbor5(param)
    graph.psfUpdate(func).get()
  }

  def initNodeFeatures(keys: Array[Long], features: Array[IntFloatVector]): Unit = {
    println(s"full batch init features")
    val param = new InitNodeFeatsParam3(graph.id, keys, features)
    val func = new InitNodeFeats3(param)
    graph.psfUpdate(func).get()
  }

  def initNodeFeatures(keys: Array[Long], features: Array[IntFloatVector],
                       numBatch: Int): Unit = {
    println(s"mini batch init features")
    println(s"keys.length=${keys.length} numBatch=${numBatch}")
    val step = keys.length / numBatch
    assert(step > 0)
    var start = 0
    while (start < keys.length) {
      val end = math.min(start + step, keys.length)
      initNodeFeatures(keys, features, start, end)
      start += step
    }
  }

  def initNodeFeatures(keys: Array[Long], features: Array[IntFloatVector],
                       start: Int, end: Int): Unit = {
    val param = new InitNodeFeatsParam4(graph.id, keys, features, start, end)
    val func = new InitNodeFeats4(param)
    graph.psfUpdate(func).get()
  }

  def getFeatures(keys: Array[Long]): Long2ObjectOpenHashMap[IntFloatVector] = {
    val func = new GetNodeFeats(new GetNodeFeatsParam(graph.id, keys.clone()))
    graph.psfGet(func).asInstanceOf[GetNodeFeatsResult].getResult
  }

  def sampleFeatures(size: Int): Array[IntFloatVector] = {
    val features = new Array[IntFloatVector](size)
    var start = 0
    var nxtSize = size
    while (start < size) {
      val func = new SampleNodeFeats(new SampleNodeFeatsParam(graph.id, nxtSize))
      val res = graph.psfGet(func).asInstanceOf[SampleNodeFeatsResult].getResult
      Array.copy(res, 0, features, start, math.min(res.length, size - start))
      start += res.length
      nxtSize = (nxtSize + 1) / 2
    }
    features
  }

  def sampleNeighbors(keys: Array[Long], count: Int): Long2ObjectOpenHashMap[Array[Long]] = {
    val func = new SampleNeighbor(new SampleNeighborParam(graph.id, keys.clone(), count))
    graph.psfGet(func).asInstanceOf[SampleNeighborResult].getNodeIdToNeighbors
  }

}

private[gcn]
object GNNPSModel {
  def apply(minId: Long, maxId: Long, weightSize: Int, optim: AsyncOptim,
            index: RDD[Long], psNumPartition: Int,
            useBalancePartition: Boolean = false): GNNPSModel = {
    val graph = new MatrixContext("graph", 1, minId, maxId)
    graph.setRowType(RowType.T_ANY_LONGKEY_SPARSE)
    graph.setValueType(classOf[Node])

    val labels = new MatrixContext("labels", 1, minId, maxId)
    labels.setRowType(RowType.T_FLOAT_SPARSE_LONGKEY)

    if (useBalancePartition)
      LoadBalancePartitioner.partition(index, maxId, psNumPartition, graph)

    val weights = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weights.setRowType(RowType.T_FLOAT_DENSE)
    weights.setPartitionerClass(classOf[ColumnRangePartitioner])

    val list = new JArrayList[MatrixContext]()
    list.add(graph)
    list.add(weights)
    list.add(labels)

    PSAgentContext.get().getMasterClient.createMatrices(list, 10000L)
    val graphId = PSAgentContext.get().getMasterClient.getMatrix("graph").getId
    val weightsId = PSAgentContext.get().getMasterClient.getMatrix("weights").getId
    val labelsId = PSAgentContext.get().getMasterClient.getMatrix("labels").getId

    new GNNPSModel(new PSMatrixImpl(graphId, 1, maxId, graph.getRowType),
      new PSVectorImpl(weightsId, 0, weights.getColNum, weights.getRowType),
      new PSVectorImpl(labelsId, 0, maxId, labels.getRowType))
  }
}
