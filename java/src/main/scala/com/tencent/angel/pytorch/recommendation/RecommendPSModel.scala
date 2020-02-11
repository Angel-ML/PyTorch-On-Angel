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
package com.tencent.angel.pytorch.recommendation

import java.io.File
import java.util.concurrent.Future
import java.util.{ArrayList => JArrayList, List => JList}

import com.tencent.angel.conf.MatrixConf
import com.tencent.angel.exception.AngelException
import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.psf.update.XavierUniform
import com.tencent.angel.ml.matrix.psf.update.base.VoidResult
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.model.output.format.{ColIdValueTextRowFormat, RowIdColIdValueTextRowFormat}
import com.tencent.angel.model.{MatrixLoadContext, MatrixSaveContext, ModelLoadContext, ModelSaveContext}
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.pytorch.model.TorchModelType
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.pytorch.recommendation.MakeUtils._
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.models.impl.{PSMatrixImpl, PSVectorImpl}
import com.tencent.angel.spark.models.{PSMatrix, PSModel, PSVector}
import it.unimi.dsi.fastutil.ints.IntOpenHashSet
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.fs.permission.FsPermission
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.{SparkContext, SparkEnv}

class RecommendPSModel(bias: PSVector,
                       weight: PSVector,
                       embedding: Option[PSMatrix],
                       mats: Option[PSVector],
                       numSlots: Int,
                       embeddingDim: Int,
                       rowType: RowType) extends Serializable {

  def getEmbeddingDim: Int = embeddingDim

  def init(): Unit = {
    if (embedding.isDefined) {
      val func = new XavierUniform(embedding.get.id, 0, embeddingDim, 1.0,
        embeddingDim, embedding.get.columns)
      embedding.get.psfUpdate(func).get()
    }

    if (mats.isDefined) {
      val func = new XavierUniform(mats.get.poolId, 0, 1, 1.0, 1, mats.get.dimension)
      mats.get.psfUpdate(func).get()
    }
  }

  // pull functions

  def getEmbedding(indices: Array[Int]): Array[Vector] =
    embedding.get.pull((0 until embeddingDim).toArray, indices)

  def getEmbedding(indices: Array[Long]): Array[Vector] =
    embedding.get.pull((0 until embeddingDim).toArray, indices)

  def getEmbedding: Array[Vector] =
    embedding.get.pull((0 until embeddingDim).toArray)

  def getMats: Vector = mats.get.pull()

  def asyncGetEmbedding(indices: Array[Int]): Future[Array[Vector]] =
    embedding.get.asyncPull((0 until embeddingDim).toArray, indices)

  def asyncGetEmbedding(indices: Array[Long]): Future[Array[Vector]] =
    embedding.get.asyncPull((0 until embeddingDim).toArray, indices)

  def asyncGetMats: Future[Vector] = mats.get.asyncPull()

  def getBiasWeight(batch: CooLongFloatMatrix, async: Boolean): (Vector, Vector) = {
    val indices = distinctIndices(batch)
    if (async) {
      val (f1, f2) = (bias.asyncPull(), weight.asyncPull(indices))
      (f1.get(), f2.get())
    } else {
      (bias.pull(), weight.pull(indices))
    }
  }

  def getBiasWeight(): (Vector, Vector) = (bias.pull(), weight.pull())

  def getBiasWeightEmbedding(batch: CooLongFloatMatrix, async: Boolean): (Vector, Vector, Array[Vector]) = {
    val indices = distinctIndices(batch)
    if (async) {
      val (f1, f2, f3) = (bias.asyncPull(), weight.asyncPull(indices), asyncGetEmbedding(indices))
      (f1.get(), f2.get(), f3.get())
    } else
      (bias.pull(), weight.pull(indices), getEmbedding(indices))
  }

  def getBiasWeightEmbedding(): (Vector, Vector, Array[Vector]) =
    (bias.pull(), weight.pull(), embedding.get.pull((0 until embeddingDim).toArray))


  def getBiasWeightEmbeddingMats(batch: CooLongFloatMatrix, async: Boolean): (Vector, Vector, Array[Vector], Vector) = {
    val indices = distinctIndices(batch)
    if (async) {
      val (f1, f2, f3, f4) = (bias.asyncPull(), weight.asyncPull(indices), asyncGetEmbedding(indices), asyncGetMats)
      (f1.get(), f2.get(), f3.get(), f4.get())
    } else
      (bias.pull(), weight.pull(indices), getEmbedding(indices), getMats)
  }

  def getBiasWeightEmbeddingMats(): (Vector, Vector, Array[Vector], Vector) = {
    val (bias, weight, embedding) = getBiasWeightEmbedding()
    (bias, weight, embedding, getMats)
  }

  // push functions
  def updateBias(grad: Vector, optim: AsyncOptim): Unit =
    optim.update(bias, 1, grad)

  def updateWeight(grad: Vector, optim: AsyncOptim): Unit =
    optim.update(weight, 1, grad)

  def updateEmbedding(grads: Array[Vector], optim: AsyncOptim): Unit =
    optim.update(embedding.get, embeddingDim, (0 until embeddingDim).toArray, grads)

  def updateMats(grad: Vector, optim: AsyncOptim): Unit =
    optim.update(mats.get, 1, grad)

  def asyncUpdateBias(grad: Vector, optim: AsyncOptim): Future[VoidResult] =
    optim.asyncUpdate(bias, 1, grad)

  def asyncUpdateWeight(grad: Vector, optim: AsyncOptim): Future[VoidResult] =
    optim.asyncUpdate(weight, 1, grad)

  def asyncUpdateEmbedding(grads: Array[Vector], optim: AsyncOptim): Future[VoidResult] =
    optim.asyncUpdate(embedding.get, embeddingDim, (0 until embeddingDim).toArray, grads)

  def asyncUpdateMats(grad: Vector, optim: AsyncOptim): Future[VoidResult] =
    optim.asyncUpdate(mats.get, 1, grad)

  def updateBiasWeight(bias: Vector, weight: Vector, optim: AsyncOptim, async: Boolean): Unit = {
    if (async) {
      val (f1, f2) = (asyncUpdateBias(bias, optim), asyncUpdateWeight(weight, optim))
      (f1.get(), f2.get())
    } else
      (updateBias(bias, optim), updateWeight(weight, optim))
  }

  def updateBiasWeightEmbedding(bias: Vector, weight: Vector, embedding: Array[Vector],
                                optim: AsyncOptim, async: Boolean): Unit = {
    if (async) {
      val (f1, f2, f3) = (asyncUpdateBias(bias, optim), asyncUpdateWeight(weight, optim),
        asyncUpdateEmbedding(embedding, optim))
      (f1.get(), f2.get(), f3.get())
    } else
      (updateBias(bias, optim), updateWeight(weight, optim),
        updateEmbedding(embedding, optim))
  }

  def updateBiasWeightEmbeddingMats(bias: Vector, weight: Vector, embedding: Array[Vector],
                                    mats: Vector, optim: AsyncOptim, async: Boolean): Unit = {
    if (async) {
      val (f1, f2, f3, f4) = (asyncUpdateBias(bias, optim), asyncUpdateWeight(weight, optim),
        asyncUpdateEmbedding(embedding, optim), asyncUpdateMats(mats, optim))
      (f1.get(), f2.get(), f3.get(), f4.get())
    } else
      (updateBias(bias, optim), updateWeight(weight, optim),
        updateEmbedding(embedding, optim),
        updateMats(mats, optim))
  }


  def distinctIndices(batch: CooLongFloatMatrix): Array[Int] = {
    val indices = new IntOpenHashSet()
    val cols = batch.getColIndices
    for (i <- cols.indices)
      indices.add(cols(i).toInt)
    indices.toIntArray
  }

  def savePSModel(output: String): Unit = {
    val ctx = new ModelSaveContext(output)

    def saveBiasWeight(): Unit = {
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val biasCtx = new MatrixSaveContext("bias", format)
      biasCtx.addIndex(0)
      ctx.addMatrix(biasCtx)

      val weightCtx = new MatrixSaveContext("weights", format)
      weightCtx.addIndex(0)
      ctx.addMatrix(weightCtx)
    }

    def saveBiasWeightEmbedding(): Unit = {
      saveBiasWeight()
      val format = classOf[RowIdColIdValueTextRowFormat].getCanonicalName
      val embeddingCtx = new MatrixSaveContext("embedding", format)
      embeddingCtx.addIndices((0 until embeddingDim).toArray)
      ctx.addMatrix(embeddingCtx)
    }

    def saveBiasWeightEmbeddingMats(): Unit = {
      saveBiasWeightEmbedding()
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val matsCtx = new MatrixSaveContext("mats", format)
      matsCtx.addIndex(0)
      ctx.addMatrix(matsCtx)
    }

    val torch = TorchModel.get()
    TorchModelType.withName(torch.getType) match {
      case TorchModelType.BIAS_WEIGHT => saveBiasWeight()
      case TorchModelType.BIAS_WEIGHT_EMBEDDING => saveBiasWeightEmbedding()
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS |
        TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        saveBiasWeightEmbeddingMats()
    }

    PSContext.instance().save(ctx)
    println(s"save angel model to $output")
    TorchModel.put(torch)
  }

  def loadPSModel(input: String): Unit = {
    val ctx = new ModelLoadContext(input)

    def loadBiasWeight(): Unit = {
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val biasCtx = new MatrixLoadContext("bias", format)
      ctx.addMatrix(biasCtx)
      val weightsCtx = new MatrixLoadContext("weights", format)
      ctx.addMatrix(weightsCtx)
    }

    def loadBiasWeightEmbedding(): Unit = {
      loadBiasWeight()
      val format = classOf[RowIdColIdValueTextRowFormat].getCanonicalName
      val embeddingCtx = new MatrixLoadContext("embedding", format)
      ctx.addMatrix(embeddingCtx)
    }

    def loadBiasWeightEmbeddingMats(): Unit = {
      loadBiasWeightEmbedding()
      val format = classOf[ColIdValueTextRowFormat].getCanonicalName
      val matsCtx = new MatrixLoadContext("mats", format)
      ctx.addMatrix(matsCtx)
    }

    val torch = TorchModel.get()
    TorchModelType.withName(torch.getType) match {
      case TorchModelType.BIAS_WEIGHT => loadBiasWeight()
      case TorchModelType.BIAS_WEIGHT_EMBEDDING => loadBiasWeightEmbedding()
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS |
        TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        loadBiasWeightEmbeddingMats()
    }
    PSContext.instance().load(ctx)
    TorchModel.put(torch)
  }

  def saveTorchModel(path: String): Unit = {
    if (rowType.isLongKey)
      throw new AngelException("cannot save torch model when using longkey")

    val torch = TorchModel.get()
    val localPath = torch.name() + "-model.pt"

    def saveBiasWeight(path: String): Unit = {
      val (bias, weight) = getBiasWeight()
      val biasInput = makeBias(bias)
      val weightInput = makeWeight(weight)
      torch.save(biasInput, weightInput, path)
    }

    def saveBiasWeightEmbedding(path: String): Unit = {
      val (bias, weight, embedding) = getBiasWeightEmbedding()
      val biasInput = makeBias(bias)
      val weightInput = makeWeight(weight)
      val embeddingInput = makeEmbedding(embedding)
      torch.save(biasInput, weightInput, embeddingInput, embeddingDim, weight.dim().toInt, path)
    }

    def saveBiasWeightEmbeddingMats(path: String): Unit = {
      val (bias, weight, embedding) = getBiasWeightEmbedding()
      val mats = getMats
      val biasInput = makeBias(bias)
      val weightInput = makeWeight(weight)
      val embeddingInput = makeEmbedding(embedding)
      val matsInput = makeMats(mats)
      torch.save(biasInput, weightInput, embeddingInput, embeddingDim, matsInput,
        torch.getMatsSize, weight.dim().toInt, path)
    }

    TorchModelType.withName(torch.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        saveBiasWeight(localPath)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        saveBiasWeightEmbedding(localPath)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS |
        TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
          saveBiasWeightEmbeddingMats(localPath)
    }

    // upload local model to hdfs
    val hdfsPath = new Path(path)
    val conf = SparkHadoopUtil.get.newConfiguration(SparkEnv.get.conf)
    val fs = hdfsPath.getFileSystem(conf)
    val outputPath = new Path(fs.makeQualified(hdfsPath).toUri.getPath)
    if (!fs.exists(outputPath)) {
      val permission = new FsPermission(FsPermission.createImmutable(0x1ff.toShort))
      FileSystem.mkdirs(fs, outputPath, permission)
    }

    val file = new File(localPath)
    if (file.exists()) {
      val srcPath = new Path(file.getPath)
      val dstPath = hdfsPath
      fs.copyFromLocalFile(srcPath, dstPath)
      println(s"save pytorch model to ${dstPath.toString}")
    }

    TorchModel.put(torch)
  }

  /* set parameters from angel to torch */
  def setParameters(torch: TorchModel): Unit = {
    TorchModelType.withName(torch.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS |
           TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>

    }
  }
}


object RecommendPSModel {

  def buildBiasWeightContexts(dim: Long, numSlots: Int, rowType: RowType, path: String): JList[MatrixContext] = {
    val bias = new MatrixContext("bias", numSlots, 1)
    bias.setRowType(RowType.T_FLOAT_DENSE)
    bias.setPartitionerClass(classOf[ColumnRangePartitioner])
    if (path.length > 0)
      bias.set(MatrixConf.MATRIX_LOAD_PATH, path)

    val weight = new MatrixContext("weights", numSlots, dim)
    weight.setRowType(rowType)
    weight.setPartitionerClass(classOf[ColumnRangePartitioner])
    if (path.length > 0)
      weight.set(MatrixConf.MATRIX_LOAD_PATH, path)

    val list = new JArrayList[MatrixContext]()
    list.add(bias)
    list.add(weight)
    list
  }

  def buildBiasWeightEmbeddingContexts(dim: Long, embeddingDim: Int,
                                       numSlots: Int,
                                       rowType: RowType,
                                       path: String): JList[MatrixContext] = {
    val list = buildBiasWeightContexts(dim, numSlots, rowType, path)
    val embedding = new MatrixContext("embedding", embeddingDim * numSlots, dim)
    embedding.setRowType(rowType)
    embedding.setPartitionerClass(classOf[ColumnRangePartitioner])
    if (path.length > 0)
      embedding.set(MatrixConf.MATRIX_LOAD_PATH, path)
    list.add(embedding)
    list
  }

  def buildBiasWeightEmbeddingMatsContexts(dim: Long, embeddingDim: Int, sizes: Array[Int],
                                           numSlots: Int, rowType: RowType, path: String): JList[MatrixContext] = {
    val list = buildBiasWeightEmbeddingContexts(dim, embeddingDim, numSlots, rowType, path)
    var sumDim = 0L
    var i = 0
    while (i < sizes.length) {
      sumDim += sizes(i) * sizes(i + 1)
      i += 2
    }

    val mats = new MatrixContext("mats", numSlots, sumDim)
    mats.setRowType(RowType.T_FLOAT_DENSE)
    mats.setPartitionerClass(classOf[ColumnRangePartitioner])
    if (path.length > 0)
      mats.set(MatrixConf.MATRIX_LOAD_PATH, path)
    list.add(mats)
    list
  }

  def buildMatrixContext(torch: TorchModel, numSlots: Int, rowType: RowType, path: String): JList[MatrixContext] = {
    TorchModelType.withName(torch.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        buildBiasWeightContexts(torch.getInputDim, numSlots, rowType, path)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        buildBiasWeightEmbeddingContexts(torch.getInputDim, torch.getEmbeddingDim, numSlots, rowType, path)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS |
           TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        buildBiasWeightEmbeddingMatsContexts(torch.getInputDim, torch.getEmbeddingDim,
          torch.getMatsSize, numSlots, rowType, path)
    }
  }

  def createModelMatrix(torch: TorchModel, numSlots: Int, rowType: RowType, path: String): JList[PSModel] = {
    val list = buildMatrixContext(torch, numSlots, rowType, path)
    val master = PSAgentContext.get().getMasterClient
    master.createMatrices(list, 10000L)

    val matrics = new JArrayList[PSModel]()
    for (i <- 0 until list.size()) {
      val ctx = list.get(i)
      val matrixId = master.getMatrix(ctx.getName).getId
      if (ctx.getName.equals("embedding"))
        matrics.add(new PSMatrixImpl(matrixId, ctx.getRowNum, ctx.getColNum, ctx.getRowType))
      else
        matrics.add(new PSVectorImpl(matrixId, 0, ctx.getColNum, ctx.getRowType))
    }
    matrics
  }

  def apply(torch: TorchModel, numSlots: Int, rowTypeStr: String, path: String): RecommendPSModel =
    apply(torch, numSlots, RowType.valueOf(rowTypeStr), path)

  def apply(torch: TorchModel, numSlots: Int, rowTypeStr: String): RecommendPSModel =
    apply(torch, numSlots, RowType.valueOf(rowTypeStr), "")

  def apply(torch: TorchModel, numSlots: Int, rowType: RowType): RecommendPSModel =
    apply(torch, numSlots, rowType, "")

  /**
    * create model matrix for bias/weight/embedding/mats
    * @param torch: model from torch
    * @param numSlots: slots for optimizer
    * @param rowType: rowType
    * @param path: passing angel model path for incremental training and predicting. Angel requires to load
    *            model partition meta when creating matrix. Passing the path to it if training from an existing
    *            model.
    * @return: RecommendPSModel
    */
  def apply(torch: TorchModel, numSlots: Int, rowType: RowType, path: String): RecommendPSModel = {
    PSContext.getOrCreate(SparkContext.getOrCreate())
    // create model
    val matrices = createModelMatrix(torch, numSlots, rowType, path)
    val bias = matrices.get(0).asInstanceOf[PSVector]
    val weight = matrices.get(1).asInstanceOf[PSVector]
    val embedding = if (matrices.size() > 2) Some(matrices.get(2).asInstanceOf[PSMatrix]) else None
    val mats = if (matrices.size() > 3) Some(matrices.get(3).asInstanceOf[PSVector]) else None
    val embeddingDim = if (matrices.size() > 2) torch.getEmbeddingDim else 0
    val model = new RecommendPSModel(bias, weight, embedding, mats, numSlots, embeddingDim, rowType)

    // load model if path.length > 0
    if (path.length > 0)
      model.loadPSModel(path)
    else // else, init it randomly
      model.init()
    model
  }
}
