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

import com.tencent.angel.graph.data.Node
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.ml.matrix.psf.update.XavierUniform
import com.tencent.angel.model.{MatrixLoadContext, MatrixSaveContext, ModelLoadContext, ModelSaveContext}
import com.tencent.angel.model.output.format.TextColumnFormat
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.utils.CheckpointUtils
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.spark.models.impl.{PSMatrixImpl, PSVectorImpl}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import org.apache.spark.rdd.RDD
import java.util.{ArrayList => JArrayList}

class SparseGNNPSModel(graph: PSMatrix,
                       weights: PSVector,
                       labels: PSVector,
                       testLabels: PSVector,
                       embedding: PSMatrix,
                       embeddingDim: Int) extends GNNPSModel(graph, weights, labels, testLabels) {

  def initEmbeddings(): Unit = {
    val func = new XavierUniform(embedding.id, 0, embeddingDim, 1.0, embeddingDim, embedding.columns)
    embedding.psfUpdate(func).get()
  }

  def getEmbedding(indices: Array[Int]): Array[Vector] = {
    embedding.pull((0 until embeddingDim).toArray, indices)
  }

  def updateEmbedding(grads: Array[Vector], optim: AsyncOptim): Unit = {
    optim.update(embedding, embeddingDim, (0 until embeddingDim).toArray, grads)
  }

  override
  def saveFeatEmbed(featEmbedPath: String): Unit = {
    val ctx = new ModelSaveContext(featEmbedPath)
    val format = classOf[TextColumnFormat].getCanonicalName
    val embeddingCtx = new MatrixSaveContext("embedding", format)
    embeddingCtx.addIndices((0 until embeddingDim).toArray)
    ctx.addMatrix(embeddingCtx)

    PSContext.instance().save(ctx)
    println(s"save feature embeddings(in the form of angel model) to $featEmbedPath.")
  }

  def loadFeatEmbed(featEmbedPath: String): Unit = {
    val ctx = new ModelLoadContext(featEmbedPath)
    val format = classOf[TextColumnFormat].getCanonicalName
    val embeddingCtx = new MatrixLoadContext("embedding", format)
    ctx.addMatrix(embeddingCtx)

    PSContext.instance().load(ctx)
  }

  override
  def checkpointMatrices(checkpointId: Int): Unit = {
    val matrixNames = new Array[String](4)
    matrixNames(0) = "graph"
    matrixNames(1) = "weights"
    matrixNames(2) = "labels"
    matrixNames(3) = "testLabels"
    matrixNames(4) = "embedding"

    CheckpointUtils.checkpoint(checkpointId, matrixNames)
  }
}

private[gcn]
object SparseGNNPSModel {
  def apply(minId: Long, maxId: Long, weightSize: Int, optim: AsyncOptim,
            index: RDD[Long], psNumPartition: Int,
            useBalancePartition: Boolean, featEmbedDim: Int, featureDim: Int): GNNPSModel = {
    var maxColBlock = if ((maxId - minId) > psNumPartition) (maxId - minId) / psNumPartition else 10
    val graph = new MatrixContext("graph", 1, minId, maxId)
    graph.setRowType(RowType.T_ANY_LONGKEY_SPARSE)
    graph.setMaxColNumInBlock(maxColBlock)
    graph.setValueType(classOf[Node])

    val labels = new MatrixContext("labels", 1, minId, maxId)
    labels.setRowType(RowType.T_FLOAT_SPARSE_LONGKEY)
    labels.setMaxColNumInBlock(maxColBlock)

    val testLabels = new MatrixContext("testLabels", 1, minId, maxId)
    testLabels.setRowType(RowType.T_FLOAT_SPARSE_LONGKEY)
    testLabels.setMaxColNumInBlock(maxColBlock)

    if (useBalancePartition)
      LoadBalancePartitioner.partition(index, psNumPartition, graph)

    maxColBlock = if (weightSize > psNumPartition) weightSize / psNumPartition else 10
    val weights = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weights.setRowType(RowType.T_FLOAT_DENSE)
    weights.setPartitionerClass(classOf[ColumnRangePartitioner])
    weights.setMaxColNumInBlock(maxColBlock)

    val embedding = new MatrixContext("embedding",
      featEmbedDim * optim.getNumSlots(), featureDim)
    embedding.setRowType(RowType.T_FLOAT_DENSE)
    embedding.setPartitionerClass(classOf[ColumnRangePartitioner])

    val list = new JArrayList[MatrixContext]()
    list.add(graph)
    list.add(weights)
    list.add(labels)
    list.add(testLabels)
    list.add(embedding)

    PSAgentContext.get().getMasterClient.createMatrices(list, 10000L)
    val graphId = PSAgentContext.get().getMasterClient.getMatrix("graph").getId
    val weightsId = PSAgentContext.get().getMasterClient.getMatrix("weights").getId
    val labelsId = PSAgentContext.get().getMasterClient.getMatrix("labels").getId
    val testLabelsId = PSAgentContext.get().getMasterClient.getMatrix("testLabels").getId
    val embeddingId = PSAgentContext.get().getMasterClient.getMatrix("embedding").getId

    new SparseGNNPSModel(new PSMatrixImpl(graphId, 1, maxId, graph.getRowType),
      new PSVectorImpl(weightsId, 0, weights.getColNum, weights.getRowType),
      new PSVectorImpl(labelsId, 0, maxId, labels.getRowType),
      new PSVectorImpl(testLabelsId, 0, maxId, testLabels.getRowType),
      new PSMatrixImpl(embeddingId, embedding.getRowNum, embedding.getColNum, embedding.getRowType),
      featEmbedDim)
  }
}