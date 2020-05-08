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

import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix
import com.tencent.angel.pytorch.data.SampleParser
import com.tencent.angel.pytorch.eval.Evaluation
import com.tencent.angel.pytorch.model.TorchModelType
import com.tencent.angel.pytorch.optim.AsyncOptim
import com.tencent.angel.pytorch.params._
import com.tencent.angel.pytorch.recommendation.MakeUtils._
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.ml.graph.params.{HasBatchSize, HasStorageLevel}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{FloatType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}

class Recommendation(torchModelPath: String, val uid: String) extends Serializable
  with HasOptimizer with HasAsync with HasNumEpoch with HasBatchSize with HasTestRatio
  with HasEvaluation with HasStorageLevel {

  def this(torchModelPath: String) = this(torchModelPath, Identifiable.randomUID("Recommendation"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  def fit(model: RecommendPSModel, data: DataFrame, optim: AsyncOptim): Unit = {
    val rdd = data.select("example").rdd.map(row => row.getString(0))
    if ($(testRatio) > 0) {
      val parts = rdd.randomSplit(Array(1 - $(testRatio), $(testRatio)))
      val (train, validate) = (parts(0), parts(1))
      fit(model, train, validate, optim)
    } else
      fit(model, rdd, null, optim)
  }

  def fit(model: RecommendPSModel, train: DataFrame, validate: DataFrame, optim: AsyncOptim): Unit = {
    val trainInput = train.select("example").rdd
      .map(row => row.getString(0))
    val validateInput = validate.select("example").rdd
      .map(row => row.getString(0))
    fit(model, trainInput, validateInput, optim)
  }

  def fit(model: RecommendPSModel, train: RDD[String], validate: RDD[String], optim: AsyncOptim): Unit = {

    train.persist($(storageLevel))
    if (validate != null)
      validate.persist($(storageLevel))

    def trainPartition(it: Iterator[String]): Iterator[(Double, Long, Long)] = {
      var (loss: Double, numBatch: Long, num: Long) = (0.0, 0L, 0L)
      val batchIterator = it.sliding($(batchSize), $(batchSize))
      while (batchIterator.hasNext) {
        val batch = batchIterator.next().toArray
        loss += optimize(batch, optim, model)
        numBatch += 1
        num += batch.length
      }
      Iterator((loss, num, numBatch))
    }

    for (epoch <- 1 to $(numEpoch)) {
      val start = System.currentTimeMillis()
      val (lossSum, num, maxBatch) = train.mapPartitions(trainPartition)
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2, math.max(f1._3, f2._3)))

      print(s"curEpoch=$epoch trainLoss=${lossSum / num} ")

      val trainMetrics = evaluate(model, train)
      trainMetrics.foreach(f => print(s"train_${f._1}=${f._2} "))
      if (validate != null) {
        val validateMetrics = evaluate(model, validate)
        validateMetrics.foreach(f => print(s"validate_${f._1}=${f._2} "))
      }

      val end = System.currentTimeMillis()
      print(s"time=${end - start}ms")
      println()

      optim.step(maxBatch.toInt)
    }
  }

  def evaluate(model: RecommendPSModel, data: RDD[String]): Map[String, Double] = {
    val scores = predict(model, data).map(f => (f._1.toFloat, f._2))
    import com.tencent.angel.pytorch.eval.Evaluation._
    Evaluation.eval(getEvaluations, scores)
  }

  def predict(model: RecommendPSModel, data: DataFrame): DataFrame = {
    val rdd = data.select("example").rdd.map(row => row.getString(0))
    val scores = predict(model, rdd)
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    val schema = StructType(Seq(
      StructField("target", StringType, nullable = false),
      StructField("score", FloatType, nullable = false)
    ))

    data.sparkSession.createDataFrame(scores, schema)
  }

  def predict(model: RecommendPSModel, data: RDD[String]): RDD[(String, Float)] = {

    def predictPartition(it: Iterator[String]): Iterator[(Array[String], Array[Float])] = {
      it.sliding($(batchSize), $(batchSize))
        .map(f => predict(f.toArray, model))
    }

    TorchModel.setPath(torchModelPath)
    data.mapPartitions(predictPartition)
      .flatMap(f => f._1.zip(f._2))

  }


  /* optimize functions */
  def optimize(batch: Array[String], optim: AsyncOptim, model: RecommendPSModel): Double = {
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val tuple3 = SampleParser.parse(batch, torch.getType)
    val (coo, fields, targets) = (tuple3._1, tuple3._2, tuple3._3)
    val loss = TorchModelType.withName(torch.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        optimizeBiasWeight(torch, model, optim, batch.length, coo, targets)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        optimizeBiasWeightEmbedding(torch, model, optim, batch.length, coo, targets)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS =>
        optimizeBiasWeightEmbeddingMats(torch, model, optim, batch.length, coo, targets, None)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        optimizeBiasWeightEmbeddingMats(torch, model, optim, batch.length, coo, targets, Some(fields))
    }

    TorchModel.put(torch)
    loss
  }

  def optimizeBiasWeight(torch: TorchModel,
                         model: RecommendPSModel,
                         optim: AsyncOptim,
                         batchSize: Int,
                         batch: CooLongFloatMatrix,
                         targets: Array[Float]): Double = {
    val (bias, weight) = model.getBiasWeight(batch, useAsync)
    val biasInput = makeBias(bias)
    val weightInput = makeWeight(weight, batch)
    val loss = torch.backward(batchSize, batch, biasInput, weightInput, targets)
    makeBiasGrad(biasInput, bias)
    makeWeightGrad(weightInput, weight, batch.getColIndices)
    model.updateBiasWeight(bias, weight, optim, useAsync)
    loss * batchSize
  }

  def optimizeBiasWeightEmbedding(torch: TorchModel,
                                  model: RecommendPSModel,
                                  optim: AsyncOptim,
                                  batchSize: Int,
                                  batch: CooLongFloatMatrix,
                                  targets: Array[Float]): Double = {
    val (bias, weight, embedding) = model.getBiasWeightEmbedding(batch, useAsync)
    val biasInput = makeBias(bias)
    val weightInput = makeWeight(weight, batch)
    val embeddingInput = makeEmbedding(embedding, batch.getColIndices, model.getEmbeddingDim)
    val loss = torch.backward(batchSize, batch, biasInput, weightInput,
      embeddingInput, model.getEmbeddingDim, targets)
    makeBiasGrad(biasInput, bias)
    makeWeightGrad(weightInput, weight, batch.getColIndices)
    makeEmbeddingGrad(embeddingInput, embedding, batch.getColIndices, model.getEmbeddingDim)
    model.updateBiasWeightEmbedding(bias, weight, embedding, optim, useAsync)
    loss * batchSize
  }

  def optimizeBiasWeightEmbeddingMats(torch: TorchModel,
                                      model: RecommendPSModel,
                                      optim: AsyncOptim,
                                      batchSize: Int,
                                      batch: CooLongFloatMatrix,
                                      targets: Array[Float],
                                      fields: Option[Array[Long]]): Double = {
    val (bias, weight, embedding, mats) = model.getBiasWeightEmbeddingMats(batch, useAsync)
    val biasInput = makeBias(bias)
    val weightInput = makeWeight(weight, batch)
    val embeddingInput = makeEmbedding(embedding, batch.getColIndices, model.getEmbeddingDim)
    val matsInput = makeMats(mats)
    val loss =
      if (fields.isEmpty)
        torch.backward(batchSize, batch, biasInput, weightInput,
          embeddingInput, model.getEmbeddingDim, matsInput, torch.getMatsSize, targets)
      else
        torch.backward(batchSize, batch, biasInput, weightInput,
          embeddingInput, model.getEmbeddingDim,
          matsInput, torch.getMatsSize,
          fields.get, targets)
    makeBiasGrad(biasInput, bias)
    makeWeightGrad(weightInput, weight, batch.getColIndices)
    makeEmbeddingGrad(embeddingInput, embedding, batch.getColIndices, model.getEmbeddingDim)
    makeMats(mats)
    model.updateBiasWeightEmbeddingMats(bias, weight, embedding, mats, optim, useAsync)
    loss * batchSize
  }

  def predict(batch: Array[String], model: RecommendPSModel): (Array[String], Array[Float]) = {
    TorchModel.setPath(torchModelPath)
    val torch = TorchModel.get()
    val tuple3 = SampleParser.parsePredict(batch, torch.getType)
    val (coo, fields, targets) = (tuple3._1, tuple3._2, tuple3._3)
    val output = TorchModelType.withName(torch.getType) match {
      case TorchModelType.BIAS_WEIGHT =>
        predictBiasWeight(torch, model, batch.length, coo)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING =>
        predictBiasWeightEmbedding(torch, model, batch.length, coo)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS =>
        predictBiasWeightEmbeddingMats(torch, model, batch.length, coo, None)
      case TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        predictBiasWeightEmbeddingMats(torch, model, batch.length, coo, Some(fields))
    }
    TorchModel.put(torch)
    (targets, output)
  }

  def predictBiasWeight(torch: TorchModel,
                        model: RecommendPSModel,
                        batchSize: Int,
                        batch: CooLongFloatMatrix): Array[Float] = {
    val (bias, weight) = model.getBiasWeight(batch, useAsync)
    val biasInput = makeBias(bias)
    val weightInput = makeWeight(weight, batch)
    torch.forward(batchSize, batch, biasInput, weightInput)
  }

  def predictBiasWeightEmbedding(torch: TorchModel,
                                 model: RecommendPSModel,
                                 batchSize: Int,
                                 batch: CooLongFloatMatrix): Array[Float] = {
    val (bias, weight, embedding) = model.getBiasWeightEmbedding(batch, useAsync)
    val biasInput = makeBias(bias)
    val weightInput = makeWeight(weight, batch)
    val embeddingInput = makeEmbedding(embedding, batch.getColIndices, model.getEmbeddingDim)
    torch.forward(batchSize, batch, biasInput, weightInput,
      embeddingInput,
      model.getEmbeddingDim)
  }

  def predictBiasWeightEmbeddingMats(torch: TorchModel,
                                     model: RecommendPSModel,
                                     batchSize: Int,
                                     batch: CooLongFloatMatrix,
                                     fields: Option[Array[Long]]): Array[Float] = {
    val (bias, weight, embedding, mats) = model.getBiasWeightEmbeddingMats(batch, useAsync)
    val biasInput = makeBias(bias)
    val weightInput = makeWeight(weight, batch)
    val embeddingInput = makeEmbedding(embedding, batch.getColIndices, model.getEmbeddingDim)
    val matsInput = makeMats(mats)
    if (fields.isEmpty)
      torch.forward(batchSize, batch, biasInput, weightInput,
        embeddingInput, model.getEmbeddingDim,
        matsInput, torch.getMatsSize)
    else
      torch.forward(batchSize, batch, biasInput, weightInput,
        embeddingInput, model.getEmbeddingDim,
        matsInput, torch.getMatsSize, fields.get)
  }

}
