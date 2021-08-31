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
package com.tencent.angel.pytorch.model

import java.io.File
import java.util.concurrent.Future
import java.util.{ArrayList => JArrayList}

import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix
import com.tencent.angel.ml.math2.storage.{IntFloatDenseVectorStorage, IntFloatSparseVectorStorage}
import com.tencent.angel.ml.math2.vector.{IntFloatVector, Vector}
import com.tencent.angel.ml.matrix.psf.update.XavierUniform
import com.tencent.angel.ml.matrix.psf.update.base.VoidResult
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.model.output.format.{ColIdValueTextRowFormat, RowIdColIdValueTextRowFormat}
import com.tencent.angel.model.{MatrixLoadContext, MatrixSaveContext, ModelLoadContext, ModelSaveContext}
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.psagent.matrix.transport.FutureResult
import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.models.impl.{PSMatrixImpl, PSVectorImpl}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import it.unimi.dsi.fastutil.ints.{Int2FloatOpenHashMap, IntOpenHashSet}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.permission.FsPermission
import org.apache.hadoop.fs.{FileSystem, Path}

import scala.collection.mutable.ArrayBuffer

class ParTorchModel(optim: AsyncOptim, path: String) extends Serializable {

  var bias: PSVector = _
  var weights: PSVector = _
  var embedding: PSMatrix = _
  var embeddings: JArrayList[PSMatrix] = _
  var mats: PSVector = _
  var params: TorchParams = _

  val biasName = "bias"
  val weightsName = "weights"
  val embeddingName = "embedding"
  val matsName = "mats"
  var embeddingsName: JArrayList[String] = _

  var useAsync = true
  var totalPullTime: Long = 0
  var totalPushTime: Long = 0
  var totalMakeParamTime: Long = 0
  var totalCalTime: Long = 0
  var totalMakeGradTime: Long = 0
  var totalCallNum: Long = 0
  var totalWaitPullTime: Long = 0
  
  var multiForwardOut: Int = 1
  
  def setMultiForwardOut(outsNum: Int): Unit = {
    this.multiForwardOut = outsNum
  }
  
  def setUseAsync(async: Boolean): Unit = {
    this.useAsync = async
  }
  
  def init(): Unit = {
    this.params = new TorchParams
    TorchModel.setPath(path)
    val torchModel = TorchModel.get()
    println(torchModel.getType)
    params.dim = torchModel.getInputDim
    TorchModelType.withName(torchModel.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        initMats(params.dim)
      
      case TorchModelType.EMBEDDINGS_MATS =>
        params.matSizes = torchModel.getMatsSize
        println("matSizes is: " + params.matSizes.mkString(","))
        initMats(params.matSizes, torchModel)
        params.inputSizes = torchModel.getInputSizes
        println("inputsSizes is: " + params.inputSizes.mkString(","))
        params.embeddingsSizes = torchModel.getEmbeddingsSize
        println("embeddingsSizes is: " + params.embeddingsSizes.mkString(","))
        initMats(params.inputSizes, params.embeddingsSizes)
      
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        params.embeddingDim = torchModel.getEmbeddingDim
        initMats(params.dim)
        initMats(params.dim, params.embeddingDim)
      
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS
           | TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        params.embeddingDim = torchModel.getEmbeddingDim
        params.numFields = torchModel.getNumFields
        params.matSizes = torchModel.getMatsSize
        println("matSizes is: " + params.matSizes.mkString(","))
        initMats(params.dim)
        initMats(params.dim, params.embeddingDim)
        initMats(params.matSizes, torchModel)
      
    }
    TorchModel.put(torchModel)
  }

  /* save pytorch module */
  def saveModule(path: String, conf: Configuration) = {
    val torchModel = TorchModel.get()
    val localPath = torchModel.name() + "-model.pt"
  
    TorchModelType.withName(torchModel.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        val biasBuf = makeBias(pullBias())
        val weightsBuf = makeWeights(pullWeight())
        torchModel.save(biasBuf, weightsBuf, localPath)
    
      case TorchModelType.EMBEDDINGS_MATS =>
        val matsBuf = makeMats(pullMats())
        val embeddingsBuf = makeEmbeddingsList(pullEmbeddingsList())
        torchModel.save(matsBuf, params.matSizes, embeddingsBuf, params.embeddingsSizes, params.inputSizes, localPath)
    
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        val biasBuf = makeBias(pullBias())
        val weightsBuf = makeWeights(pullWeight())
        val embeddingBuf = makeEmbeddings(pullEmbeddings())
        torchModel.save(biasBuf, weightsBuf, embeddingBuf,
          params.embeddingDim, params.dim.toInt, localPath)
    
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS
           | TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        val biasBuf = makeBias(pullBias())
        val weightsBuf = makeWeights(pullWeight())
        val embeddingBuf = makeEmbeddings(pullEmbeddings())
        val matsBuf = makeMats(pullMats())
        torchModel.save(biasBuf, weightsBuf, embeddingBuf,
          params.embeddingDim, matsBuf, params.matSizes, params.dim.toInt, localPath)
    
    }
  
    //save to hdfs
    var savedModulePath = new Path(path)
    val fs = savedModulePath.getFileSystem(conf)
    savedModulePath = fs.makeQualified(savedModulePath)
    savedModulePath = new Path(savedModulePath.toUri.getPath)
    val sysPerms = new FsPermission(FsPermission.createImmutable(0x1ff.toShort))
    FileSystem.mkdirs(fs, savedModulePath, sysPerms)
    val file = new File(localPath)
    if (file.exists()) {
      val srcPath = new Path(file.getPath)
      val dstPath = new Path(path)
      fs.copyFromLocalFile(srcPath, dstPath)
    }
    TorchModel.put(torchModel)
  }


  def initEmbedding(): Unit = {
    embedding.psfUpdate(new XavierUniform(embedding.id, 0, params.embeddingDim, 1.0,
      embedding.rows, embedding.columns)).get()
  }

  def initEmbeddings(): Unit = {
    for(i <- 0 until embeddings.size()) {
      val embedding = embeddings.get(i)
      embedding.psfUpdate(new XavierUniform(embedding.id, 0, params.embeddingsSizes(i), 1.0,
        embedding.rows, embedding.columns)).get()
    }
  }

  def initMats(): Unit =
    mats.psfUpdate(new XavierUniform(mats.poolId, 0, 1, 1.0, 1, mats.dimension)).get()

  def initMats(inputDim: Long): Unit = {
    val biasCtx = new MatrixContext(biasName, optim.getNumSlots(), 1)
    biasCtx.setRowType(RowType.T_FLOAT_DENSE)
    biasCtx.setPartitionerClass(classOf[ColumnRangePartitioner])

    val weightCtx = new MatrixContext(weightsName, optim.getNumSlots(), inputDim)
    weightCtx.setRowType(RowType.T_FLOAT_DENSE)
    weightCtx.setPartitionerClass(classOf[ColumnRangePartitioner])

    val list = new JArrayList[MatrixContext]()
    list.add(biasCtx)
    list.add(weightCtx)

    val master = PSAgentContext.get().getMasterClient
    master.createMatrices(list, Long.MaxValue)
    bias = new PSVectorImpl(master.getMatrix(biasCtx.getName).getId, 0,
      biasCtx.getColNum, biasCtx.getRowType)
    weights = new PSVectorImpl(master.getMatrix(weightCtx.getName).getId, 0,
      weightCtx.getColNum, weightCtx.getRowType)
  }

  def initMats(inputDim: Long, embeddingDim: Int): Unit = {
    val embeddingCtx = new MatrixContext(embeddingName, embeddingDim * optim.getNumSlots(), inputDim)
    embeddingCtx.setRowType(RowType.T_FLOAT_DENSE)
    embeddingCtx.setPartitionerClass(classOf[ColumnRangePartitioner])

    val master = PSAgentContext.get().getMasterClient
    master.createMatrix(embeddingCtx, Long.MaxValue)
    embedding = new PSMatrixImpl(master.getMatrix(embeddingCtx.getName).getId, embeddingCtx.getRowNum,
      embeddingCtx.getColNum, embeddingCtx.getRowType)
    initEmbedding()
  }

  def initMats(inputSizes: Array[Long], embeddingsSize: Array[Int]): Unit = {
    val mcSizes = inputSizes.length
    val matrixContexts: JArrayList[MatrixContext] = new JArrayList[MatrixContext](mcSizes)
    embeddingsName = new JArrayList[String](mcSizes)
    embeddings = new JArrayList[PSMatrix](mcSizes)
    for(i <- 0 until mcSizes) {
      val embeddingName = "embedding_" + i
      val embeddingCtx = new MatrixContext(embeddingName,  embeddingsSize(i) * optim.getNumSlots(), inputSizes(i))
      embeddingCtx.setRowType(RowType.T_FLOAT_DENSE)
      embeddingCtx.setPartitionerClass(classOf[ColumnRangePartitioner])
      embeddingsName.add(embeddingName)
      matrixContexts.add(embeddingCtx)
    }
    val master = PSAgentContext.get().getMasterClient
    master.createMatrices(matrixContexts, Long.MaxValue)
    for(i <- 0 until matrixContexts.size()) {
      val embeddingCtx = matrixContexts.get(i)
      embeddings.add(new PSMatrixImpl(master.getMatrix(embeddingCtx.getName).getId, embeddingCtx.getRowNum,
        embeddingCtx.getColNum, embeddingCtx.getRowType))
    }
    initEmbeddings()
  }

  def initMats(matSizes: Array[Int]): Unit = {
    var sumDim = 0L
    var i = 0
    while (i < matSizes.length) {
      sumDim += matSizes(i) * matSizes(i + 1)
      i += 2
    }

    val matCtx = new MatrixContext(matsName, optim.getNumSlots(), sumDim)
    matCtx.setPartitionerClass(classOf[ColumnRangePartitioner])
    matCtx.setRowType(RowType.T_FLOAT_DENSE)
    val master = PSAgentContext.get().getMasterClient
    master.createMatrix(matCtx, Long.MaxValue)
    val matId = master.getMatrix(matCtx.getName).getId
    mats = new PSVectorImpl(matId, 0, matCtx.getColNum, matCtx.getRowType)
    initMats()
  }
  
  def initMats(matSizes: Array[Int], torchModel: TorchModel): Unit = {
    var sumDim = 0L
    var i = 0
    while (i < matSizes.length) {
      sumDim += matSizes(i) * matSizes(i + 1)
      i += 2
    }
    
    val matCtx = new MatrixContext(matsName, optim.getNumSlots(), sumDim)
    matCtx.setPartitionerClass(classOf[ColumnRangePartitioner])
    matCtx.setRowType(RowType.T_FLOAT_DENSE)
    val master = PSAgentContext.get().getMasterClient
    master.createMatrix(matCtx, Long.MaxValue)
    val matId = master.getMatrix(matCtx.getName).getId
    mats = new PSVectorImpl(matId, 0, matCtx.getColNum, matCtx.getRowType)
    initMats(torchModel)
  }
  
  def initMats(torchModel: TorchModel): Unit = {
    val update = VFactory.denseFloatVector(torchModel.getMatsParameters)
    mats.update(update)
  }

  /* pull functions */

  def pullWeightBias(indices: Array[Int], useAsync: Boolean): (IntFloatVector, IntFloatVector) = {
    if (useAsync) {
      val weightFuture = asyncPullWeight(indices)
      val biasFuture = asyncPullBias()
      (weightFuture.get.asInstanceOf[IntFloatVector], biasFuture.get.asInstanceOf[IntFloatVector])
    } else {
      (pullWeight(indices), pullBias())
    }
  }
  
  def pullMatsEmbeddings(indices: Array[Int], useAsync: Boolean): (IntFloatVector, Array[Array[IntFloatVector]]) = {
    if (useAsync) {
      val matsFuture = asyncPullMats()
      val embeddingsListFuture = asyncPullEmbeddingsList(indices)
      val embeddingsList = new ArrayBuffer[Array[IntFloatVector]]()
      for (embeddingFuture: Future[Array[Vector]] <- embeddingsListFuture) {
        embeddingsList.append(embeddingFuture.get().map(vectorFuture => vectorFuture.asInstanceOf[IntFloatVector]))
      }
      (matsFuture.get().asInstanceOf[IntFloatVector], embeddingsList.toArray)
    } else {
      (pullMats(), pullEmbeddingsList(indices))
    }
  }

  def pullWeightBiasEmbedding(indices: Array[Int], useAsync: Boolean): (IntFloatVector, IntFloatVector, Array[IntFloatVector]) = {
    if (useAsync) {
      val weightFuture = asyncPullWeight(indices)
      val biasFuture = asyncPullBias()
      val embeddingFuture = asyncPullEmbeddings(indices)

      (weightFuture.get.asInstanceOf[IntFloatVector], biasFuture.get.asInstanceOf[IntFloatVector],
        embeddingFuture.get().map(vectorFuture => vectorFuture.asInstanceOf[IntFloatVector]))
    } else {
      (pullWeight(indices), pullBias(), pullEmbeddings(indices))
    }
  }

  def pullWeightBiasEmbeddingMats(indices: Array[Int], useAsync: Boolean): (IntFloatVector, IntFloatVector, Array[IntFloatVector], IntFloatVector) = {
    if (useAsync) {
      val weightFuture = asyncPullWeight(indices)
      val biasFuture = asyncPullBias()
      val embeddingFuture = asyncPullEmbeddings(indices)
      val matsFuture = asyncPullMats()

      (weightFuture.get.asInstanceOf[IntFloatVector], biasFuture.get.asInstanceOf[IntFloatVector],
        embeddingFuture.get().map(vectorFuture => vectorFuture.asInstanceOf[IntFloatVector]),
        matsFuture.get().asInstanceOf[IntFloatVector])
    } else {
      (pullWeight(indices), pullBias(), pullEmbeddings(indices), pullMats())
    }
  }

  def pullBias(): IntFloatVector =
    bias.pull().asInstanceOf[IntFloatVector]

  def pullWeight(indices: Array[Int]): IntFloatVector =
    weights.pull(indices).asInstanceOf[IntFloatVector]

  def pullWeight(): IntFloatVector =
    weights.pull().asInstanceOf[IntFloatVector]

  def pullEmbeddings(indices: Array[Int]): Array[IntFloatVector] = {
    val rows = (0 until params.embeddingDim).toArray
    return embedding.pull(rows, indices).map(f => f.asInstanceOf[IntFloatVector])
  }
  
  def pullEmbeddingsList(indices: Array[Int]): Array[Array[IntFloatVector]] = {
    val embeddingsList = new ArrayBuffer[Array[IntFloatVector]]()
    for (i <- 0 until embeddings.size()) {
      val embedding = embeddings.get(i)
      val rows = (0 until params.embeddingsSizes(i)).toArray
      embeddingsList.append(embedding.pull(rows, indices).map(f => f.asInstanceOf[IntFloatVector]))
    }
    embeddingsList.toArray
  }

  def pullEmbeddingsList(): Array[Array[IntFloatVector]] = {
    val embeddingsList = new ArrayBuffer[Array[IntFloatVector]]()
    for(i <- 0 until embeddings.size()) {
      val embedding = embeddings.get(i)
      val rows = (0 until params.embeddingsSizes(i)).toArray
      embeddingsList.append(embedding.pull(rows).map(f => f.asInstanceOf[IntFloatVector]))
    }
    embeddingsList.toArray
  }

  def pullEmbeddings(): Array[IntFloatVector] = {
    val rows = (0 until params.embeddingDim).toArray
    return embedding.pull(rows).map(f => f.asInstanceOf[IntFloatVector])
  }
  
  def pullMats(): IntFloatVector =
    mats.pull().asInstanceOf[IntFloatVector]

  def asyncPullBias(): Future[Vector] =
    bias.asyncPull()

  def asyncPullWeight(indices: Array[Int]): Future[Vector] =
    weights.asyncPull(indices)

  def asyncPullEmbeddings(indices: Array[Int]): Future[Array[Vector]] = {
    val rows = (0 until params.embeddingDim).toArray
    embedding.asyncPull(rows, indices)
  }

  def asyncPullEmbeddingsList(indices: Array[Int]): Array[Future[Array[Vector]]] = {
    val embeddingsFutureList = new ArrayBuffer[Future[Array[Vector]]]()
    for(i <- 0 until embeddings.size()) {
      val embedding = embeddings.get(i)
      val rows = (0 until params.embeddingsSizes(i)).toArray
      embeddingsFutureList.append(embedding.asyncPull(rows, indices))
    }
    embeddingsFutureList.toArray
  }

  def asyncPullMats(): Future[Vector] =
    mats.asyncPull()

  /* push functions */
  def pushWeightBias(weightGrad: IntFloatVector, biasGrad: IntFloatVector, useAsync: Boolean): Unit = {
    if (useAsync) {
      asyncPushWeight(weightGrad)
      asyncPushBias(biasGrad)
    } else {
      pushWeight(weightGrad)
      pushBias(biasGrad)
    }
  }

  def pushWeightBiasMats(weightGrad: IntFloatVector, biasGrad: IntFloatVector, matsGrad: IntFloatVector, useAsync: Boolean): Unit = {
    if (useAsync) {
      asyncPushWeight(weightGrad)
      asyncPushBias(biasGrad)
      asyncPushMats(matsGrad)
    } else {
      pushWeight(weightGrad)
      pushBias(biasGrad)
      pushMats(matsGrad)
    }
  }

  def pushEmbeddingsMats(matsGrad: IntFloatVector, embeddingsListGrads: Array[Array[IntFloatVector]], useAsync: Boolean): Unit = {
    if (useAsync) {
      asyncPushMats(matsGrad)
      asyncPushEmbeddingsList(embeddingsListGrads)
    } else {
      pushMats(matsGrad)
      pushEmbeddingsList(embeddingsListGrads)
    }
  }

  def pushWeightBiasEmbedding(weightGrad: IntFloatVector, biasGrad: IntFloatVector,
                              embeddingGrads: Array[IntFloatVector], useAsync: Boolean): Unit = {
    if (useAsync) {
      asyncPushWeight(weightGrad)
      asyncPushBias(biasGrad)
      asyncPushEmbedding(embeddingGrads)
    } else {
      pushWeight(weightGrad)
      pushBias(biasGrad)
      pushEmbedding(embeddingGrads)
    }
  }

  def pushWeightBiasEmbeddingMats(weightGrad: IntFloatVector, biasGrad: IntFloatVector,
                                  embeddingGrads: Array[IntFloatVector], matsGrad: IntFloatVector,
                                  useAsync: Boolean) = {
    if (useAsync) {
      asyncPushWeight(weightGrad)
      asyncPushBias(biasGrad)
      asyncPushEmbedding(embeddingGrads)
      asyncPushMats(matsGrad)
    } else {
      pushWeight(weightGrad)
      pushBias(biasGrad)
      pushEmbedding(embeddingGrads)
      pushMats(matsGrad)
    }
  }

  def pushBias(grad: IntFloatVector): Unit =
    optim.update(bias, 1, grad)

  def pushWeight(grad: IntFloatVector): Unit =
    optim.update(weights, 1, grad)

  def pushEmbedding(grads: Array[IntFloatVector]): Unit = {
    val rows = (0 until params.embeddingDim).toArray
    optim.update(embedding, params.embeddingDim, rows, grads.map(f => f.asInstanceOf[Vector]))
  }
  
  def pushEmbeddingsList(grads: Array[Array[IntFloatVector]]): Unit = {
    for (index <- grads.indices) {
      val rows = (0 until params.embeddingsSizes(index)).toArray
      optim.update(embeddings.get(index), params.embeddingsSizes(index), rows, grads(index).map(f => f.asInstanceOf[Vector]))
    }
  }
  
  def pushMats(grad: IntFloatVector): Unit =
    optim.update(mats, 1, grad)

  def asyncPushBias(grad: IntFloatVector): Future[VoidResult] =
    optim.asyncUpdate(bias, 1, grad)

  def asyncPushWeight(grad: IntFloatVector): Future[VoidResult] =
    optim.asyncUpdate(weights, 1, grad)

  def asyncPushEmbedding(grads: Array[IntFloatVector]): Future[VoidResult] = {
    val rows = (0 until params.embeddingDim).toArray
    optim.asyncUpdate(embedding, params.embeddingDim, rows, grads.map(f => f.asInstanceOf[Vector]))
  }
  
  def asyncPushEmbeddingsList(grads: Array[Array[IntFloatVector]]): Future[VoidResult] = {
    for (index <- grads.indices) {
      val rows = (0 until params.embeddingsSizes(index)).toArray
      optim.asyncUpdate(embeddings.get(index), params.embeddingsSizes(index), rows, grads(index).map(f => f.asInstanceOf[Vector]))
    }
    new FutureResult[VoidResult]
  }

  def asyncPushMats(grad: IntFloatVector): Future[VoidResult] =
    optim.asyncUpdate(mats, 1, grad)

  /* making pytorch parameters/gradients functions */
  def makeBias(bias: IntFloatVector): Array[Float] = {
    val buf = new Array[Float](1)
    buf(0) = bias.get(0)
    buf
  }

  def makeBiasGrad(biasBuf: Array[Float], bias: IntFloatVector): Unit =
    bias.set(0, biasBuf(0))

  def makeWeights(weight: IntFloatVector, feats: Array[Long]): Array[Float] = {
    val buf = new Array[Float](feats.length)
    for (i <- 0 until feats.length)
      buf(i) = weight.get(feats(i).toInt)
    buf
  }

  def makeWeights(weight: IntFloatVector): Array[Float] = {
    val buf = new Array[Float](weight.size())
    for (i <- 0 until weight.size())
      buf(i) = weight.get(i)
    buf
  }

  def makeWeightsGrad(weightsBuf: Array[Float], weights: IntFloatVector, feats: Array[Long]): Unit = {
    val grad = new Int2FloatOpenHashMap(weights.size())
    for (i <- 0 until weightsBuf.length)
      grad.addTo(feats(i).toInt, weightsBuf(i))
    weights.setStorage(new IntFloatSparseVectorStorage(weights.dim().toInt, grad))
  }

  def makeEmbeddings(embeddings: Array[IntFloatVector], feats: Array[Long]): Array[Float] = {
    val buf = new Array[Float](feats.length * params.embeddingDim)
    for (i <- 0 until feats.length)
      for (j <- 0 until params.embeddingDim)
        buf(i * params.embeddingDim + j) = embeddings(j).get(feats(i).toInt)
    buf
  }
  
  def makeEmbeddingsList(embeddingsList: Array[Array[IntFloatVector]], feats: Array[Long]): Array[Array[Float]] = {
    val bufList = new ArrayBuffer[Array[Float]]()
    val size = embeddingsList.length
    for (index <- 0 until size) {
      val embeddingDim = params.embeddingsSizes(index)
      val buf = new Array[Float](feats.length * embeddingDim)
      for (i <- 0 until feats.length)
        for (j <- 0 until embeddingDim)
          buf(i * embeddingDim + j) = embeddingsList(index)(j).get(feats(i).toInt)
      bufList.append(buf)
    }
    bufList.toArray
  }

  def makeEmbeddingsList(embeddingsList: Array[Array[IntFloatVector]]): Array[Array[Float]] = {
    val bufList = new ArrayBuffer[Array[Float]]()
    val size = embeddingsList.length
    for(index <- 0 until size) {
      val embeddingDim = params.embeddingsSizes(index)
      val featsLength = embeddingsList(index)(0).size()
      val buf = new Array[Float](featsLength * embeddingDim)
      for (i <- 0 until featsLength)
        for (j <- 0 until embeddingDim)
          buf(i * embeddingDim + j) = embeddingsList(index)(j).get(i)
      bufList.append(buf)
    }
    bufList.toArray
  }

  def makeEmbeddings(embeddings: Array[IntFloatVector]): Array[Float] = {
    val buf = new Array[Float](embeddings(0).size() * params.embeddingDim)
    for (i <- 0 until embeddings(0).size())
      for (j <- 0 until params.embeddingDim)
        buf(i * params.embeddingDim + j) = embeddings(j).get(i)
    buf
  }

  def makeEmbeddingGrad(embeddingBuf: Array[Float], embedding: Array[IntFloatVector], feats: Array[Long]): Unit = {
    val grads = embedding.map(f => f.size()).map(f => new Int2FloatOpenHashMap(f))
    for (i <- 0 until feats.length) {
      for (j <- 0 until params.embeddingDim) {
        grads(j).addTo(feats(i).toInt, embeddingBuf(i * params.embeddingDim + j))
      }
    }

    embedding.zip(grads).foreach {
      case (e, g) =>
        e.setStorage(new IntFloatSparseVectorStorage(e.dim().toInt, g))
    }
  }

  def makeEmbeddingsListGrad(embeddingBuf: Array[Array[Float]], embedding: Array[Array[IntFloatVector]], feats: Array[Long]): Unit = {
    for(index <- embeddingBuf.indices) {
      val grads = embedding(index).map(f => f.size()).map(f => new Int2FloatOpenHashMap(f))
      for (i <- 0 until feats.length) {
        for (j <- 0 until params.embeddingsSizes(index)) {
          grads(j).addTo(feats(i).toInt, embeddingBuf(index)(i * params.embeddingsSizes(index) + j))
        }
      }
      embedding(index).zip(grads).foreach {
        case (e, g) =>
          e.setStorage(new IntFloatSparseVectorStorage(e.dim().toInt, g))
      }
    }
  }
  
  def makeEmbeddingsListGrad(rawEmbeddingBuf: Array[Array[Float]], embedding: Array[Array[IntFloatVector]],
                             feats: Array[Long], batchSize: Int): Unit = {
    val embeddingBuf = rawEmbeddingBuf.map(arr => arr.map(_ / batchSize))
    for (index <- embeddingBuf.indices) {
      val grads = embedding(index).map(f => f.size()).map(f => new Int2FloatOpenHashMap(f))
      for (i <- 0 until feats.length) {
        for (j <- 0 until params.embeddingsSizes(index)) {
          grads(j).addTo(feats(i).toInt, embeddingBuf(index)(i * params.embeddingsSizes(index) + j))
        }
      }
      embedding(index).zip(grads).foreach {
        case (e, g) =>
          e.setStorage(new IntFloatSparseVectorStorage(e.dim().toInt, g))
      }
    }
  }
  
  def makeMats(mats: IntFloatVector): Array[Float] =
    mats.getStorage.asInstanceOf[IntFloatDenseVectorStorage].getValues

  def makeMatsGrad(matsBuf: Array[Float], mats: IntFloatVector): Unit =
    mats.setStorage(new IntFloatDenseVectorStorage(matsBuf))
  
  def makeMatsGrad(matsBuf: Array[Float], mats: IntFloatVector, batchSize: Int): Unit = {
    val avgBuf = matsBuf.map(_ / batchSize)
    mats.setStorage(new IntFloatDenseVectorStorage(avgBuf))
  }
  
  /* get pull indices functions */
  def distinctIntIndices(batch: CooLongFloatMatrix): Array[Int] = {
    val indices = new IntOpenHashSet()
  
    val cols = batch.getColIndices
    for (i <- 0 until cols.length)
      indices.add(cols(i).toInt)
  
    return indices.toIntArray
  }

  /* optimize functions */
  def optimize(batch: Array[String]): Double = {
    TorchModel.setPath(path)
    val torchModel = TorchModel.get()
    val tuple3 = SampleParser.parse(batch, torchModel.getType)
    val (coo, fields, targets) = (tuple3._1, tuple3._2, tuple3._3)
  
    val loss = TorchModelType.withName(torchModel.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        optimizeBiasWeight(torchModel, batch.length, coo, targets)
      case TorchModelType.EMBEDDINGS_MATS =>
        optimizeEmbeddingsMats(torchModel, batch.length, coo, targets)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        optimizeBiasWeightEmbedding(torchModel, batch.length, coo, targets)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS =>
        optimizeBiasWeightEmbeddingMats(torchModel, batch.length, coo, targets)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        optimizeBiasWeightEmbeddingMatsField(torchModel, batch.length, coo, fields, targets)
    }
    TorchModel.put(torchModel)
    loss
  }

  def optimizeBiasWeight(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix, targets: Array[Float]): Double = {
    val indices = distinctIntIndices(batch)

    incCallNum
    var start = 0L

    // Pull parameters
    start = System.currentTimeMillis()
    val (weights, bias) = pullWeightBias(indices, useAsync)
    incPullTime(start)

    // Transfer the parameters formats from angel to pytorch
    start = System.currentTimeMillis()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    incMakeParamTime(start)

    // Calculate the gradients
    start = System.currentTimeMillis()
    val loss = torchModel.backward(batchSize, batch, biasBuf, weightsBuf, targets)

    incCalTime(start)

    // Transfer the parameters formats from pytorch to angel
    start = System.currentTimeMillis()
    makeBiasGrad(biasBuf, bias)
    makeWeightsGrad(weightsBuf, weights, batch.getColIndices)
    incCalTime(start)

    // Push the gradient to ps
    start = System.currentTimeMillis()
    pushWeightBias(weights, bias, useAsync)
    incPushTime(start)

    //    println(s"avgPullTime=${avgPullTime} avgMakeParamTime=${avgMakeParamTime} gradTime=${avgCalTime} " +
    //      s"avgMakeGradTime=${avgMakeGradTime} avgPushTime=${avgPushTime} loss=$loss")

    return loss * batchSize
  }
  
  def optimizeEmbeddingsMats(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix, targets: Array[Float]): Double = {
    val indices = distinctIntIndices(batch)
    
    incCallNum
    var start = 0L
    
    // Pull parameters
    start = System.currentTimeMillis()
    val (mats, embeddings) = pullMatsEmbeddings(indices, useAsync)
    incPullTime(start)
    
    // Transfer the parameters formats from angel to pytorch
    start = System.currentTimeMillis()
    val matsBuf = makeMats(mats)
    
    val embeddingsListBuf = makeEmbeddingsList(embeddings, batch.getColIndices)
    incMakeParamTime(start)
    
    // Calculate the gradients
    start = System.currentTimeMillis()
    val loss = torchModel.backward(batchSize, batch, matsBuf, params.matSizes, embeddingsListBuf, params.embeddingsSizes, targets)
    incCalTime(start)
    
    // Transfer the parameters formats from pytorch to angel
    start = System.currentTimeMillis()
    makeMatsGrad(matsBuf, mats, batchSize)
    makeEmbeddingsListGrad(embeddingsListBuf, embeddings, batch.getColIndices, batchSize)
    incMakeGradTime(start)
    
    // Push the gradient to ps
    start = System.currentTimeMillis()
    pushEmbeddingsMats(mats, embeddings, useAsync)
    incPushTime(start)
    
    println(s"avgPullTime=${avgPullTime} avgMakeParamTime=${avgMakeParamTime} gradTime=${avgCalTime} " +
      s"avgMakeGradTime=${avgMakeGradTime} avgPushTime=${avgPushTime} loss=$loss")
    
    loss * batchSize
  }

  def optimizeBiasWeightEmbedding(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix, targets: Array[Float]): Double = {
    val indices = distinctIntIndices(batch)

    incCallNum
    var start = 0L

    // Pull parameters
    start = System.currentTimeMillis()
    val (weights, bias, embedding) = pullWeightBiasEmbedding(indices, useAsync)
    incPullTime(start)

    // Transfer the parameters formats from angel to pytorch
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    incMakeParamTime(start)

    // Calculate the gradients
    start = System.currentTimeMillis()
    val loss = torchModel.backward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf,
      params.embeddingDim, targets)
    incCalTime(start)

    // Transfer the parameters formats from pytorch to angel
    start = System.currentTimeMillis()
    makeBiasGrad(biasBuf, bias)
    makeWeightsGrad(weightsBuf, weights, batch.getColIndices)
    makeEmbeddingGrad(embeddingBuf, embedding, batch.getColIndices)
    incMakeGradTime(start)

    // Push the gradient to ps
    start = System.currentTimeMillis()
    pushWeightBiasEmbedding(weights, bias, embedding, useAsync)
    incPushTime(start)

    //    println(s"avgPullTime=$avgPullTime avgMakeParamTime=${avgMakeParamTime} gradTime=${avgCalTime} " +
    //      s"avgMakeGradTime=$avgMakeGradTime avgPushTime=${avgPushTime} loss=$loss")

    return loss * batchSize
  }

  def optimizeBiasWeightEmbeddingMats(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix, targets: Array[Float]): Double = {
    val indices = distinctIntIndices(batch)

    incCallNum
    var start = 0L

    // Pull parameters
    start = System.currentTimeMillis()
    val (weights, bias, embedding, mats) = pullWeightBiasEmbeddingMats(indices, useAsync)
    incPullTime(start)

    // Transfer the parameters formats from angel to pytorch
    start = System.currentTimeMillis()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    val matsBuf = makeMats(mats)
    incMakeParamTime(start)

    // Calculate the gradients
    start = System.currentTimeMillis()
    val loss = torchModel.backward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf,
      params.embeddingDim, matsBuf, params.matSizes, targets)
    incCalTime(start)

    // Transfer the parameters formats from pytorch to angel
    start = System.currentTimeMillis()
    makeBiasGrad(biasBuf, bias)
    makeWeightsGrad(weightsBuf, weights, batch.getColIndices)
    makeEmbeddingGrad(embeddingBuf, embedding, batch.getColIndices)
    makeMatsGrad(matsBuf, mats)
    incMakeGradTime(start)

    // Push the gradient to ps
    start = System.currentTimeMillis()
    pushWeightBiasEmbeddingMats(weights, bias, embedding, mats, useAsync)
    incPushTime(start)

    println(s"avgPullTime=${avgPullTime} avgMakeParamTime=${avgMakeParamTime} gradTime=${avgCalTime} " +
      s"avgMakeGradTime=${avgMakeGradTime} avgPushTime=${avgPushTime} loss=$loss")

    return loss * batchSize
  }

  def optimizeBiasWeightEmbeddingMatsField(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix, fields: Array[Long], targets: Array[Float]): Double = {
    var (start, end) = (0L, 0L)
    val indices = distinctIntIndices(batch)
    start = System.currentTimeMillis()
    val bias = pullBias()
    val weights = pullWeight(indices)
    val embedding = pullEmbeddings(indices)
    val mats = pullMats()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    val matsBuf = makeMats(mats)
    end = System.currentTimeMillis()
    val pullTime = end - start

    start = System.currentTimeMillis()
    val loss = torchModel.backward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf,
      params.embeddingDim, matsBuf, params.matSizes, fields, targets)
    end = System.currentTimeMillis()
    val gradTime = end - start

    start = System.currentTimeMillis()
    makeBiasGrad(biasBuf, bias)
    makeWeightsGrad(weightsBuf, weights, batch.getColIndices)
    makeEmbeddingGrad(embeddingBuf, embedding, batch.getColIndices)
    makeMatsGrad(matsBuf, mats)
    pushBias(bias)
    pushWeight(weights)
    pushEmbedding(embedding)
    pushMats(mats)
    end = System.currentTimeMillis()

    val pushTime = end - start
    println(s"pullTime=${pullTime} gradTime=${gradTime} pushTime=${pushTime} loss=$loss")
    return loss * batchSize
  }

  /* predict functions */
  
  def predict(batch: Array[String]): (Array[String], Array[Float]) = {
    TorchModel.setPath(path)
    val torchModel = TorchModel.get()
    val tuple3 = SampleParser.parsePredict(batch, torchModel.getType)
    val (coo, fields, targets) = (tuple3._1, tuple3._2, tuple3._3)
    
    TorchModelType.withName(torchModel.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        TorchModel.put(torchModel)
        return (targets, predictBiasWeight(torchModel, batch.length, coo))
      case TorchModelType.EMBEDDINGS_MATS =>
        TorchModel.put(torchModel)
        return (targets, predictEmbeddingsMats(torchModel, batch.length, coo, this.multiForwardOut))
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        TorchModel.put(torchModel)
        return (targets, predictBiasWeightEmbedding(torchModel, batch.length, coo))
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS =>
        TorchModel.put(torchModel)
        return (targets, predictBiasWeightEmbeddingMats(torchModel, batch.length, coo))
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        TorchModel.put(torchModel)
        return (targets, predictBiasWeightEmbeddingMatsField(torchModel, batch.length, coo, fields))
    }
    TorchModel.put(torchModel)
    return null
  }

  def predictBiasWeight(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix): Array[Float] = {
    val indices = distinctIntIndices(batch)
    val bias = pullBias()
    val weights = pullWeight(indices)
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    return torchModel.forward(batchSize, batch, biasBuf, weightsBuf)
  }
  
  def predictEmbeddingsMats(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix, multiFOut: Int = 1): Array[Float] = {
    val indices = distinctIntIndices(batch)
    val mats = pullMats()
    val matsBuf = makeMats(mats)
    val embeddingsList = pullEmbeddingsList(indices)
    val embeddingsListBuf = makeEmbeddingsList(embeddingsList, batch.getColIndices)
    torchModel.forward(batchSize, batch, matsBuf, params.matSizes, embeddingsListBuf, params.embeddingsSizes, multiFOut)
  }

  def predictBiasWeightEmbedding(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix): Array[Float] = {
    val indices = distinctIntIndices(batch)
    val bias = pullBias()
    val weights = pullWeight(indices)
    val embedding = pullEmbeddings(indices)
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    return torchModel.forward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf, params.embeddingDim)
  }

  def predictBiasWeightEmbeddingMats(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix): Array[Float] = {
    val indices = distinctIntIndices(batch)
    val bias = pullBias()
    val weights = pullWeight(indices)
    val embedding = pullEmbeddings(indices)
    val mats = pullMats()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    val matsBuf = makeMats(mats)
    return torchModel.forward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf, params.embeddingDim,
      matsBuf, params.matSizes)
  }

  def predictBiasWeightEmbeddingMatsField(torchModel: TorchModel, batchSize: Int, batch: CooLongFloatMatrix, fields: Array[Long]): Array[Float] = {
    val indices = distinctIntIndices(batch)
    val bias = pullBias()
    val weights = pullWeight(indices)
    val embedding = pullEmbeddings(indices)
    val mats = pullMats()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    val matsBuf = makeMats(mats)
    return torchModel.forward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf, params.embeddingDim,
      matsBuf, params.matSizes, fields)
  }


  /* save/load model to/from angel format */
  def save(output: String): Unit = {
    val modelSaveCtx = new ModelSaveContext(output)
  
    def addBiasCtx(): Unit = {
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val biasCtx = new MatrixSaveContext(biasName, format)
      biasCtx.addIndex(0)
      modelSaveCtx.addMatrix(biasCtx)
    }
  
    def addWeightsCtx(): Unit = {
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val weightsCtx = new MatrixSaveContext(weightsName, format)
      weightsCtx.addIndex(0)
      modelSaveCtx.addMatrix(weightsCtx)
    }
  
    def addEmbeddingCtx(): Unit = {
      val format = classOf[RowIdColIdValueTextRowFormat].getCanonicalName
      val embeddingCtx = new MatrixSaveContext(embeddingName, format)
      embeddingCtx.addIndices((0 until params.embeddingDim).toArray)
      modelSaveCtx.addMatrix(embeddingCtx)
    }
  
    def addEmbeddingsCtx(): Unit = {
      for (i <- 0 until embeddingsName.size()) {
        val format = classOf[RowIdColIdValueTextRowFormat].getCanonicalName
        val embeddingCtx = new MatrixSaveContext(embeddingsName.get(i), format)
        embeddingCtx.addIndices((0 until params.embeddingsSizes(i)).toArray)
        modelSaveCtx.addMatrix(embeddingCtx)
      }
    }
  
    def addMatsCtx(): Unit = {
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val matsCtx = new MatrixSaveContext(matsName, format)
      matsCtx.addIndex(0)
      modelSaveCtx.addMatrix(matsCtx)
    }
  
    TorchModel.setPath(path)
    val torchModel = TorchModel.get()
    TorchModelType.withName(torchModel.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        addBiasCtx()
        addWeightsCtx()
      case TorchModelType.EMBEDDINGS_MATS =>
        addMatsCtx()
        addEmbeddingsCtx()
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        addBiasCtx()
        addWeightsCtx()
        addEmbeddingCtx()
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS
           | TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        addBiasCtx()
        addWeightsCtx()
        addEmbeddingCtx()
        addMatsCtx()
    }
  
    PSContext.instance().save(modelSaveCtx)
    TorchModel.put(torchModel)
  }
  
  def load(input: String): Unit = {
    val modelLoadCtx = new ModelLoadContext(input)
    
    def addBiasCtx(): Unit = {
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val biasCtx = new MatrixLoadContext(biasName, format)
      modelLoadCtx.addMatrix(biasCtx)
    }
    
    def addWeightsCtx(): Unit = {
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val weightsCtx = new MatrixLoadContext(weightsName, format)
      modelLoadCtx.addMatrix(weightsCtx)
    }
    
    def addEmbeddingCtx(): Unit = {
      val format = classOf[RowIdColIdValueTextRowFormat].getCanonicalName
      val embeddingCtx = new MatrixLoadContext(embeddingName, format)
      modelLoadCtx.addMatrix(embeddingCtx)
    }
    
    def addEmbeddingsCtx(): Unit = {
      for (i <- 0 until embeddingsName.size()) {
        val format = classOf[RowIdColIdValueTextRowFormat].getCanonicalName
        val embeddingCtx = new MatrixLoadContext(embeddingsName.get(i), format)
        modelLoadCtx.addMatrix(embeddingCtx)
      }
    }
    
    def addMatsCtx(): Unit = {
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val matsCtx = new MatrixLoadContext(matsName, format)
      modelLoadCtx.addMatrix(matsCtx)
    }
    
    TorchModel.setPath(path)
    val torchModel = TorchModel.get()
    TorchModelType.withName(torchModel.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        addBiasCtx()
        addWeightsCtx()
      case TorchModelType.EMBEDDINGS_MATS =>
        addMatsCtx()
        addEmbeddingsCtx()
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        addBiasCtx()
        addWeightsCtx()
        addEmbeddingCtx()
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS |
           TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        addBiasCtx()
        addWeightsCtx()
        addEmbeddingCtx()
        addMatsCtx()
    }
    
    PSContext.instance().load(modelLoadCtx)
    TorchModel.put(torchModel)
  }

  /* time calculate functions */
  def incPullTime(startTs: Long): Unit = {
    totalPullTime = totalPullTime + (System.currentTimeMillis() - startTs)
  }

  def incPushTime(startTs: Long): Unit = {
    totalPushTime = totalPushTime + (System.currentTimeMillis() - startTs)
  }

  def incMakeParamTime(startTs: Long): Unit = {
    totalMakeParamTime = totalMakeParamTime + (System.currentTimeMillis() - startTs)
  }

  def incCalTime(startTs: Long): Unit = {
    totalCalTime = totalCalTime + (System.currentTimeMillis() - startTs)
  }

  def incMakeGradTime(startTs: Long): Unit = {
    totalMakeGradTime = totalMakeGradTime + (System.currentTimeMillis() - startTs)
  }

  def incCallNum(): Unit = {
    totalCallNum = totalCallNum + 1
  }

  def avgPullTime: Long = {
    totalPullTime / totalCallNum
  }

  def avgPushTime: Long = {
    totalPushTime / totalCallNum
  }

  def avgMakeParamTime: Long = {
    totalMakeParamTime / totalCallNum
  }

  def avgMakeGradTime: Long = {
    totalMakeGradTime / totalCallNum
  }

  def avgCalTime: Long = {
    totalCalTime / totalCallNum
  }

}
