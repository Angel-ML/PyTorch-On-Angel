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

import com.tencent.angel.graph.data.GraphNode
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.ml.matrix.psf.update.XavierUniform
import com.tencent.angel.model.{MatrixLoadContext, MatrixSaveContext, ModelLoadContext, ModelSaveContext}
import com.tencent.angel.model.output.format.TextColumnFormat
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.utils.CheckpointUtils
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import org.apache.spark.rdd.RDD
import com.tencent.angel.graph.common.param.ModelContext
import com.tencent.angel.graph.utils.ModelContextUtils
import org.apache.spark.SparkContext

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
    val userNumNode = index.distinct().count()
    //create user graph matrix context
    val graphModelContext = new ModelContext(psNumPartition, minId, maxId, userNumNode, "graph",
      SparkContext.getOrCreate().hadoopConfiguration)
    val graphMatrixContext = ModelContextUtils.createMatrixContext(graphModelContext,
      RowType.T_ANY_LONGKEY_SPARSE, classOf[GraphNode])

    // create labels matrix context
    val labelsModelContext = new ModelContext(psNumPartition, minId, maxId, userNumNode, "labels",
      SparkContext.getOrCreate().hadoopConfiguration)
    val labelsMatrixContext = ModelContextUtils.createMatrixContext(labelsModelContext, RowType.T_FLOAT_SPARSE_LONGKEY)

    // create testLabels matrix context
    val testLabelsModelContext = new ModelContext(psNumPartition, minId, maxId, userNumNode,
      "testLabels", SparkContext.getOrCreate().hadoopConfiguration)
    val testLabelsMatrixContext = ModelContextUtils.createMatrixContext(testLabelsModelContext, RowType.T_FLOAT_SPARSE_LONGKEY)

    if (!graphModelContext.isUseHashPartition && useBalancePartition)
      LoadBalancePartitioner.partition(index, psNumPartition, graphMatrixContext)

    // create weights matrix context
    val weights = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weights.setRowType(RowType.T_FLOAT_DENSE)
    weights.setPartitionerClass(classOf[ColumnRangePartitioner])

    // create user embedding matrix context
    val embedding = new MatrixContext("embedding",
      featEmbedDim * optim.getNumSlots(), featureDim)
    embedding.setRowType(RowType.T_FLOAT_DENSE)
    embedding.setPartitionerClass(classOf[ColumnRangePartitioner])

    // create matrix
    val graph = PSMatrix.matrix(graphMatrixContext)
    val weightsVec = PSVector.vector(weights)
    val labels = PSVector.vector(labelsMatrixContext)
    val testLabels = PSVector.vector(testLabelsMatrixContext)
    val embeddingMatrix = PSMatrix.matrix(embedding)

    new SparseGNNPSModel(graph, weightsVec, labels, testLabels, embeddingMatrix, featEmbedDim)
  }
}