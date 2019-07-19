package com.tencent.angel.pytorch.graph

import com.tencent.angel.graph.data.Node
import com.tencent.angel.ml.math2.vector.{IntFloatVector, LongFloatVector}
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import org.apache.spark.rdd.RDD
import java.util.{ArrayList => JArrayList}

import com.tencent.angel.graph.client.getnodefeats2.GetNodeFeatsResult
import com.tencent.angel.graph.client.getnodefeats2.{GetNodeFeats, GetNodeFeatsParam}
import com.tencent.angel.graph.client.initNeighbor4.{InitNeighbor, InitNeighborParam}
import com.tencent.angel.graph.client.initnodefeats2.InitNodeFeatsParam
import com.tencent.angel.graph.client.initnodefeats2.InitNodeFeats
import com.tencent.angel.graph.client.sampleneighbor3.SampleNeighborResult
import com.tencent.angel.graph.client.sampleneighbor3.{SampleNeighbor, SampleNeighborParam}
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.pytorch.optim.{AsyncAdam, AsyncOptim}
import com.tencent.angel.spark.models.impl.{PSMatrixImpl, PSVectorImpl}
import com.tencent.angel.spark.util.VectorUtils
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap

private[graph]
class GraphPSModel(graph: PSMatrix,
                   weights: PSVector,
                   labels: PSVector) extends Serializable {

  val dim: Long = labels.dimension

  def readLabels(keys: Array[Long]): LongFloatVector =
    labels.pull(keys.clone()).asInstanceOf[LongFloatVector]

  def setLabels(value: LongFloatVector): Unit =
    labels.update(value)

  def numNodes(): Long =
    VectorUtils.nnz(labels)

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

  def initNeighbours(keys: Array[Long],
                     indptr: Array[Int],
                     neighbours: Array[Long]): Unit = {
    val func = new InitNeighbor(new InitNeighborParam(graph.id, keys, indptr, neighbours))
    graph.psfUpdate(func).get()
  }

  def initNodeFeatures(keys: Array[Long], features: Array[IntFloatVector]): Unit = {
//    println(s"keys:${keys.mkString(",")}")
    val func = new InitNodeFeats(new InitNodeFeatsParam(graph.id, keys.clone(), features))
    graph.psfUpdate(func).get()
  }

  def getFeatures(keys: Array[Long]): Long2ObjectOpenHashMap[IntFloatVector] = {
    val func = new GetNodeFeats(new GetNodeFeatsParam(graph.id, keys.clone()))
    graph.psfGet(func).asInstanceOf[GetNodeFeatsResult].getResult
  }

  def sampleNeighbors(keys: Array[Long], count: Int): Long2ObjectOpenHashMap[Array[Long]] = {
    val func = new SampleNeighbor(new SampleNeighborParam(graph.id, keys.clone(), count))
    graph.psfGet(func).asInstanceOf[SampleNeighborResult].getNodeIdToNeighbors
  }

}

private[graph]
object GraphPSModel {
  def apply(minId: Long, maxId: Long, weightSize: Int, optim: AsyncOptim,
            index: RDD[Long], psNumPartition: Int,
            useBalancePartition: Boolean = false): GraphPSModel = {
    val graph = new MatrixContext("graph", 1, minId, maxId)
    graph.setRowType(RowType.T_ANY_LONGKEY_SPARSE)
    graph.setValueType(classOf[Node])

    val labels = new MatrixContext("labels", 1, minId, maxId)
    labels.setRowType(RowType.T_FLOAT_SPARSE_LONGKEY)

    if (useBalancePartition)
      LoadBalancePartitioner.partition(index, maxId, psNumPartition, graph)

    val weights = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weights.setRowType(RowType.T_FLOAT_DENSE)

    val list = new JArrayList[MatrixContext]()
    list.add(graph)
    list.add(weights)
    list.add(labels)

    PSAgentContext.get().getMasterClient.createMatrices(list, 10000L)
    val graphId = PSAgentContext.get().getMasterClient.getMatrix("graph").getId
    val weightsId = PSAgentContext.get().getMasterClient.getMatrix("weights").getId
    val labelsId = PSAgentContext.get().getMasterClient.getMatrix("labels").getId

    new GraphPSModel(new PSMatrixImpl(graphId, 1, maxId, graph.getRowType),
      new PSVectorImpl(weightsId, 0, weights.getColNum, weights.getRowType),
      new PSVectorImpl(labelsId, 0, maxId, labels.getRowType))
  }
}
