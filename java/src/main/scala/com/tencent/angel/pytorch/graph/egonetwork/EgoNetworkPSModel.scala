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
package com.tencent.angel.pytorch.graph.egonetwork

import java.util.{ArrayList => JArrayList, Arrays => JArrays}

import com.tencent.angel.graph.client.initNeighbor5.{InitNeighbor, InitNeighborParam}
import com.tencent.angel.graph.client.sampleneighbor3.{SampleNeighbor, SampleNeighborParam, SampleNeighborResult}
import com.tencent.angel.graph.data.Node
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.vector.LongIntVector
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.spark.models.impl.{PSMatrixImpl, PSVectorImpl}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import org.apache.spark.rdd.RDD

private[egonetwork]
class EgoNetworkPSModel(graph: PSMatrix,
                        nodes: PSVector, // Only contains the labeled keys
                        srcs: PSVector)  // Only contains the first neighbors of the labeled keys
  extends Serializable {

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long]): Unit = {
    val func = new InitNeighbor(new InitNeighborParam(graph.id, keys, indptr, neighbors))
    graph.psfUpdate(func).get()
  }

  def initNeighbors(keys: Array[Long],
                    indptr: Array[Int],
                    neighbors: Array[Long],
                    numBatch: Int): Unit = {
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
    val param = new InitNeighborParam(graph.id, keys, indptr, neighbors, start, end)
    val func = new InitNeighbor(param)
    graph.psfUpdate(func).get()
  }

  def sampleNeighbors(keys: Array[Long], count: Int): Long2ObjectOpenHashMap[Array[Long]] = {
    val func = new SampleNeighbor(new SampleNeighborParam(graph.id, keys.clone(), count))
    graph.psfGet(func).asInstanceOf[SampleNeighborResult].getNodeIdToNeighbors
  }

  def initNodesWithLabels(keys: Array[Long]): Unit = {
    val values = Array.fill(keys.length)(1)
    JArrays.sort(keys)
    val update = VFactory.sortedLongKeyIntVector(nodes.dimension, keys, values)
    nodes.update(update)
  }

  def readNodes(keys: Array[Long]): LongIntVector =
    nodes.pull(keys).asInstanceOf[LongIntVector]

  def setNodes(keys: Array[Long]): Unit = {
    val values = Array.fill(keys.length)(1)
    JArrays.sort(keys)
    val update = VFactory.sortedLongKeyIntVector(nodes.dimension, keys, values)
    nodes.update(update)
  }

  def updateSrcKeys(keys: Array[Long]): Unit = {
    val values = Array.fill(keys.length)(1)
    JArrays.sort(keys)
    val update = VFactory.sortedLongKeyIntVector(srcs.dimension, keys, values)
    srcs.update(update)
  }

  def readSrcKeys(keys: Array[Long]): LongIntVector =
    srcs.pull(keys).asInstanceOf[LongIntVector]

  def readALLSrcKeys(): LongIntVector =
    srcs.pull().asInstanceOf[LongIntVector]
}

private[egonetwork]
object EgoNetworkPSModel {
  def apply(minId: Long, maxId: Long, index: RDD[Long],
            psNumPartition: Int,
            useBalancePartition: Boolean = false): EgoNetworkPSModel = {
    val graph = new MatrixContext("graph", 1, minId, maxId)
    graph.setRowType(RowType.T_ANY_LONGKEY_SPARSE)
    graph.setValueType(classOf[Node])

    val nodes = new MatrixContext("nodes", 1, minId, maxId)
    nodes.setRowType(RowType.T_INT_SPARSE_LONGKEY)
    nodes.setValidIndexNum(-1)

    val srcs = new MatrixContext("srcs", 1, minId, maxId)
    srcs.setRowType(RowType.T_INT_SPARSE_LONGKEY)
    srcs.setValidIndexNum(-1)

    if (useBalancePartition)
      LoadBalancePartitioner.partition(index, maxId, psNumPartition, graph)

    val list = new JArrayList[MatrixContext]()
    list.add(graph)
    list.add(nodes)
    list.add(srcs)

    PSAgentContext.get().getMasterClient.createMatrices(list, 1000L)
    val graphId = PSAgentContext.get().getMasterClient.getMatrix("graph").getId
    val nodesId = PSAgentContext.get().getMasterClient.getMatrix("nodes").getId
    val srcIds = PSAgentContext.get().getMasterClient.getMatrix("srcs").getId
    new EgoNetworkPSModel(new PSMatrixImpl(graphId, 1, maxId, graph.getRowType),
      new PSVectorImpl(nodesId, 0, maxId, nodes.getRowType),
      new PSVectorImpl(srcIds, 0, maxId, srcs.getRowType))
  }
}
