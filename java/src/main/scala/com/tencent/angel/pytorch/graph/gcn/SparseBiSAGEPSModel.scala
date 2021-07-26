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

import com.tencent.angel.graph.common.param.ModelContext
import com.tencent.angel.graph.data.GraphNode
import com.tencent.angel.graph.utils.ModelContextUtils
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.psf.update.XavierUniform
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.model.{MatrixLoadContext, MatrixSaveContext, ModelLoadContext, ModelSaveContext}
import com.tencent.angel.model.output.format.TextColumnFormat
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.pytorch.utils.CheckpointUtils
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer


class SparseBiSAGEPSModel(userGraph: PSMatrix,
                          itemGraph: PSMatrix,
                          weights: PSVector,
                          labels: PSVector,
                          testLabels: PSVector,
                          userEmbedding: PSMatrix,
                          userEmbeddingDim: Int,
                          itemEmbedding: PSMatrix = null,
                          itemEmbeddingDim: Int = -1) extends
  BiSAGEPSModel(userGraph, itemGraph, weights, labels, testLabels) {

  def initUserEmbeddings(): Unit = {
    val func = new XavierUniform(userEmbedding.id, 0, userEmbeddingDim, 1.0,
      userEmbeddingDim, userEmbedding.columns)
    userEmbedding.psfUpdate(func).get()
  }

  def initItemEmbeddings(): Unit = {
    val func = new XavierUniform(itemEmbedding.id, 0, itemEmbeddingDim, 1.0,
      itemEmbeddingDim, itemEmbedding.columns)
    itemEmbedding.psfUpdate(func).get()
  }

  def getEmbedding(indices: Array[Int], graphType: Int = 0): Array[Vector] = {
    if (graphType == 0)
      userEmbedding.pull((0 until userEmbeddingDim).toArray, indices)
    else
      itemEmbedding.pull((0 until itemEmbeddingDim).toArray, indices)
  }

  def updateUserEmbedding(grads: Array[Vector], optim: AsyncOptim): Unit = {
    optim.update(userEmbedding, userEmbeddingDim, (0 until userEmbeddingDim).toArray, grads)
  }

  def updateItemEmbedding(grads: Array[Vector], optim: AsyncOptim): Unit = {
    optim.update(itemEmbedding, itemEmbeddingDim, (0 until itemEmbeddingDim).toArray, grads)
  }

  override
  def saveFeatEmbed(featEmbedPath: String): Unit = {
    val ctx = new ModelSaveContext(featEmbedPath)
    val format = classOf[TextColumnFormat].getCanonicalName
    val userEmbeddingCtx = new MatrixSaveContext("userEmbedding", format)
    userEmbeddingCtx.addIndices((0 until userEmbeddingDim).toArray)
    ctx.addMatrix(userEmbeddingCtx)

    if (itemEmbedding != null) {
      val itemEmbeddingCtx = new MatrixSaveContext("itemEmbedding", format)
      itemEmbeddingCtx.addIndices((0 until itemEmbeddingDim).toArray)
      ctx.addMatrix(itemEmbeddingCtx)
    }

    PSContext.instance().save(ctx)
    println(s"save user (and item) feature embeddings(in the form of angel model) to $featEmbedPath.")
  }

  def loadFeatEmbed(featEmbedPath: String): Unit = {
    val ctx = new ModelLoadContext(featEmbedPath)
    val format = classOf[TextColumnFormat].getCanonicalName
    val userEmbeddingCtx = new MatrixLoadContext("userEmbedding", format)
    ctx.addMatrix(userEmbeddingCtx)

    if (itemEmbedding != null){
      val itemEmbeddingCtx = new MatrixLoadContext("itemEmbedding", format)
      ctx.addMatrix(itemEmbeddingCtx)
    }

    PSContext.instance().load(ctx)
  }

  override
  def checkpointMatrices(checkpointId: Int): Unit = {
    val matrixNames = new ArrayBuffer[String]()
    matrixNames.append("userGraph")
    matrixNames.append("itemGraph")
    matrixNames.append("weights")
    matrixNames.append("labels")
    matrixNames.append("testLabels")
    matrixNames.append("userEmbedding")
    if (itemEmbedding != null) matrixNames.append("itemEmbedding")
    CheckpointUtils.checkpoint(checkpointId, matrixNames.toArray)
  }
}

private[gcn]
object SparseBiSAGEPSModel {
  def apply(userMinId: Long, userMaxId: Long, itemMinId: Long, itemMaxId: Long, weightSize: Int, optim: AsyncOptim,
            index: RDD[Long], psNumPartition: Int, useBalancePartition: Boolean = false, userFeatEmbedDim: Int,
            userFeatureDim: Int, itemFeatEmbedDim: Int, itemFeatureDim: Int): SparseBiSAGEPSModel = {
    val userNumNode = index.distinct().count()
    //create user graph matrix context
    val userModelContext = new ModelContext(psNumPartition, userMinId, userMaxId, userNumNode, "userGraph",
      SparkContext.getOrCreate().hadoopConfiguration)
    val userGraphMatrixContext = ModelContextUtils.createMatrixContext(userModelContext,
      RowType.T_ANY_LONGKEY_SPARSE, classOf[GraphNode])

    // create labels matrix context
    val labelsModelContext = new ModelContext(psNumPartition, userMinId, userMaxId, userNumNode, "labels",
      SparkContext.getOrCreate().hadoopConfiguration)
    val labelsMatrixContext = ModelContextUtils.createMatrixContext(labelsModelContext, RowType.T_FLOAT_SPARSE_LONGKEY)

    // create testLabels matrix context
    val testLabelsModelContext = new ModelContext(psNumPartition, userMinId, userMaxId, userNumNode,
      "testLabels", SparkContext.getOrCreate().hadoopConfiguration)
    val testLabelsMatrixContext = ModelContextUtils.createMatrixContext(testLabelsModelContext, RowType.T_FLOAT_SPARSE_LONGKEY)

    // create item graph matrix context
    val itemModelContext = new ModelContext(psNumPartition, itemMinId, itemMaxId, -1, "itemGraph",
      SparkContext.getOrCreate().hadoopConfiguration)
    val itemGraphMatrixContext = ModelContextUtils.createMatrixContext(itemModelContext,
      RowType.T_ANY_LONGKEY_SPARSE, classOf[GraphNode])

    if (!userModelContext.isUseHashPartition && useBalancePartition)
      LoadBalancePartitioner.partition(index, psNumPartition, userGraphMatrixContext)

    // create weights matrix context
    val weights = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weights.setRowType(RowType.T_FLOAT_DENSE)
    weights.setPartitionerClass(classOf[ColumnRangePartitioner])

    // create user embedding matrix context
    val userEmbedding = new MatrixContext("userEmbedding",
      userFeatEmbedDim * optim.getNumSlots(), userFeatureDim)
    userEmbedding.setRowType(RowType.T_FLOAT_DENSE)
    userEmbedding.setPartitionerClass(classOf[ColumnRangePartitioner])

    // create item embedding matrix context
    var itemEmbedding: MatrixContext = null
    if (itemFeatureDim > 0) {
      itemEmbedding = new MatrixContext("itemEmbedding",
        itemFeatEmbedDim * optim.getNumSlots(), itemFeatureDim)
      itemEmbedding.setRowType(RowType.T_FLOAT_DENSE)
      itemEmbedding.setPartitionerClass(classOf[ColumnRangePartitioner])
    }

    // create matrix
    val userGraph = PSMatrix.matrix(userGraphMatrixContext)
    val itemGraph = PSMatrix.matrix(itemGraphMatrixContext)
    val weightsVec = PSVector.vector(weights)
    val labels = PSVector.vector(labelsMatrixContext)
    val testLabels = PSVector.vector(testLabelsMatrixContext)
    val userEmbeddingMatrix = PSMatrix.matrix(userEmbedding)
    val itemEmbeddingMatrix = if (itemEmbedding != null) PSMatrix.matrix(itemEmbedding) else null

    new SparseBiSAGEPSModel(userGraph, itemGraph, weightsVec, labels, testLabels, userEmbeddingMatrix,
      userFeatEmbedDim, itemEmbeddingMatrix, itemFeatEmbedDim)
  }
}