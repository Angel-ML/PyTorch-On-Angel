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
package com.tencent.angel.pytorch.embedding

import java.text.SimpleDateFormat
import java.util.Date
import java.util.concurrent.TimeUnit

import com.tencent.angel.ml.core.optimizer.decayer.{StandardDecay, StepSizeScheduler}
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.model.{MatrixLoadContext, MatrixSaveContext, ModelLoadContext, ModelSaveContext}
import com.tencent.angel.pytorch.torch.TorchModel
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.embedding.FastSigmoid
import com.tencent.angel.spark.ml.embedding.line2._
import com.tencent.angel.spark.models.PSMatrix
import it.unimi.dsi.fastutil.ints.{Int2IntOpenHashMap, Int2ObjectOpenHashMap}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class Word2VecModel(numNode: Int,
                    embeddingDim: Int,
                    numPart: Int,
                    maxLength: Int,
                    torchModelPath: String,
                    numNodesPerRow: Int,
                    seed: Int = Random.nextInt,
                    learningRate: Float,
                    decayRate: Float) extends Serializable {

  val matrixName = "embedding"
  // Create one ps matrix to hold the input vectors and the output vectors for all node
  val mc: MatrixContext = new MatrixContext(matrixName, 1, numNode)
  mc.setMaxRowNumInBlock(1)
  mc.setMaxColNumInBlock(numNode / numPart)
  mc.setRowType(RowType.T_ANY_INTKEY_DENSE)
  mc.setValueType(classOf[LINENode])
  mc.setInitFunc(new LINEInitFunc(2, embeddingDim))
  val psMatrix: PSMatrix = PSMatrix.matrix(mc)
  val matrixId: Int = psMatrix.id
  var totalPullTime: Long = 0
  var totalPushTime: Long = 0
  var totalMakeParamTime: Long = 0
  var totalCalTime: Long = 0
  var totalMakeGradTime: Long = 0
  var totalCallNum: Long = 0
  var totalWaitPullTime: Long = 0
  val ssScheduler: StepSizeScheduler = new StandardDecay(learningRate, decayRate)

  // initialize embeddings
  def randomInitialize(seed: Int): Unit = {
    val beforeRandomize = System.currentTimeMillis()
    psMatrix.psfUpdate(new LINEModelRandomize(new RandomizeUpdateParam(matrixId, embeddingDim, 2, seed))).get()
    logTime(s"Model successfully Randomized, cost ${(System.currentTimeMillis() - beforeRandomize) / 1000.0}s")
  }

  def this(params: Word2VecParam) {
    this(params.maxIndex, params.embeddingDim, params.numPSPart, params.maxLength, params.torchModelPath, params.nodesNumPerRow, params.seed,
      params.learningRate, params.decayRate)
  }

  def train(corpus: RDD[Array[Int]], params: Word2VecParam, path: String): Unit = {
    var learningRate = params.learningRate
    for (epoch <- 1 to params.numEpoch) {
      val epochStartTime = System.currentTimeMillis()
      val (lossSum, size) = corpus.mapPartitions {
        iterator =>
          iterator.sliding(params.batchSize, params.batchSize)
            .map(batch => (optimize(batch.toArray, params.windowSize, params.negSample, params.maxIndex, learningRate), batch.length))
      }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
      learningRate = ssScheduler.next().toFloat
      val epochTime = System.currentTimeMillis() - epochStartTime
      println(s"epoch=$epoch loss=${lossSum / size} time=${epochTime.toFloat / 1000}s")
      if (epoch % params.checkpointInterval == 0 && epoch < params.numEpoch) {
        logTime(s"Epoch=${epoch}, checkpoint the model")
        val startTs = System.currentTimeMillis()
        psMatrix.checkpoint(epoch)
        logTime(s"checkpoint use time=${System.currentTimeMillis() - startTs}")
      }

      if (epoch % params.saveModelInterval == 0 && epoch < params.numEpoch) {
        logTime(s"Epoch=${epoch}, save the model")
        val startTs = System.currentTimeMillis()
        save(path, epoch)
        logTime(s"save use time=${System.currentTimeMillis() - startTs}")
      }
    }
  }

  def save(modelPathRoot: String, epoch: Int): Unit = {
    save(new Path(modelPathRoot, s"CP_$epoch").toString)
  }

  def save(modelPath: String): Unit = {
    logTime(s"saving model to $modelPath")
    val ss = SparkSession.builder().getOrCreate()
    deleteIfExists(modelPath, ss)

    val saveContext = new ModelSaveContext(modelPath)
    saveContext.addMatrix(new MatrixSaveContext(matrixName, classOf[TextLINEModelOutputFormat].getTypeName))
    PSContext.instance().save(saveContext)
  }

  def load(modelPath: String): Unit = {
    val startTime = System.currentTimeMillis()
    logTime(s"load model from $modelPath")

    val loadContext = new ModelLoadContext(modelPath)
    loadContext.addMatrix(new MatrixLoadContext(psMatrix.name))
    PSContext.getOrCreate(SparkContext.getOrCreate()).load(loadContext)
    logTime(s"model load time=${System.currentTimeMillis() - startTime} ms")
  }

  private def deleteIfExists(modelPath: String, ss: SparkSession): Unit = {
    val path = new Path(modelPath)
    val fs = path.getFileSystem(ss.sparkContext.hadoopConfiguration)
    if (fs.exists(path)) {
      fs.delete(path, true)
    }
  }

  def optimize(batch: Array[Array[Int]], windowSize: Int, numNegSample: Int, maxIndex: Int, learningRate: Float): Double = {
    //TorchModel.setPath(torchModelPath)
    //val torchModel = TorchModel.get()
    val getBatchDataStartTime = System.currentTimeMillis()
    val trainBatch = Word2VecModel.parseBatchData(batch, windowSize, numNegSample, maxIndex)
    println(s"Get batch data cost time=${(System.currentTimeMillis()-getBatchDataStartTime).toFloat / 1000}s")
    //val loss = optimizeOneBatch(torchModel, batch.length, trainBatch._1, trainBatch._2, trainBatch._3, numNegSample, learningRate)
    val loss = optimizeOneBatchRaw(batch.length, trainBatch._1, trainBatch._2, trainBatch._3, numNegSample, learningRate)
    //TorchModel.put(torchModel)
    loss
  }

  def optimizeOneBatch(torchModel: TorchModel, batchSize: Int, srcNodes: Array[Int], dstNodes: Array[Int], negativeSamples: Array[Array[Int]],
                       numNegSample: Int, learningRate: Float): Double = {
    incCallNum()
    var start = 0L
    start = System.currentTimeMillis()
    println("w2v test, srcNodes length is: " + srcNodes.length)
    val result =  psMatrix.asyncPsfGet(new LINEGetEmbedding(new LINEGetEmbeddingParam(matrixId, srcNodes, dstNodes,
      negativeSamples, 2, numNegSample))).get(1800000, TimeUnit.MILLISECONDS).asInstanceOf[LINEGetEmbeddingResult].getResult
    val srcFeats: Int2ObjectOpenHashMap[Array[Float]] = result._1
    val dstFeats: Int2ObjectOpenHashMap[Array[Float]] = result._2
    incPullTime(start)

    // Transfer the parameters formats from angel to pytorch
    start = System.currentTimeMillis()
    val (srcEmbeddings, dstEmbeddings, negativeEmbeddings) = makeWord2vecEmbeddings(srcNodes, dstNodes, negativeSamples, srcFeats, dstFeats, numNegSample)
    incMakeParamTime(start)
    // Calculate the gradients
    start = System.currentTimeMillis()
    var loss = torchModel.word2VecBackward(srcNodes.length, negativeSamples.length*numNegSample, srcEmbeddings, dstEmbeddings, negativeEmbeddings, embeddingDim)
    incCalTime(start)
    // Transfer the parameters formats from pytorch to angel
    start = System.currentTimeMillis()
    val (inputUpdates, outputUpdates) = makeWord2VecGrad(srcEmbeddings, dstEmbeddings, negativeEmbeddings, srcNodes, dstNodes, negativeSamples, srcFeats.size(),
      dstFeats.size(), numNegSample, learningRate)
    incMakeGradTime(start)
    // Push the gradient to ps
    start = System.currentTimeMillis()
    psMatrix.psfUpdate(new LINEAdjust(new LINEAdjustParam(matrixId, inputUpdates, outputUpdates, 2)))
    incPushTime(start)

    loss = loss / (srcNodes.length*(numNegSample+1))
    println(s"avgPullTime=$avgPullTime avgMakeParamTime=$avgMakeParamTime gradTime=$avgCalTime " +
      s"avgMakeGradTime=$avgMakeGradTime avgPushTime=$avgPushTime loss=$loss")

    loss * batchSize
  }

  def optimizeOneBatchRaw(batchSize: Int, srcNodes: Array[Int], dstNodes: Array[Int], negativeSamples: Array[Array[Int]],
                       numNegSample: Int, learningRate: Float): Double = {
    incCallNum()
    var start = 0L
    start = System.currentTimeMillis()
    println("w2v test, srcNodes length is: " + srcNodes.length)
    val result =  psMatrix.asyncPsfGet(new LINEGetEmbedding(new LINEGetEmbeddingParam(matrixId, srcNodes, dstNodes,
      negativeSamples, 2, numNegSample))).get(1800000, TimeUnit.MILLISECONDS).asInstanceOf[LINEGetEmbeddingResult].getResult
    val srcFeats: Int2ObjectOpenHashMap[Array[Float]] = result._1
    val dstFeats: Int2ObjectOpenHashMap[Array[Float]] = result._2
    incPullTime(start)

    // Calculate the gradients
    start = System.currentTimeMillis()
    val dots = dot(srcNodes, dstNodes, negativeSamples, srcFeats, dstFeats, numNegSample)
    var loss = doGrad(dots, numNegSample, learningRate)
    incCalTime(start)
    start = System.currentTimeMillis()
    val (inputUpdates, outputUpdates) = adjust(srcNodes, dstNodes, negativeSamples, srcFeats, dstFeats, numNegSample, dots)
    incMakeGradTime(start)
    // Push the gradient to ps
    start = System.currentTimeMillis()
    psMatrix.psfUpdate(new LINEAdjust(new LINEAdjustParam(matrixId, inputUpdates, outputUpdates, 2)))
    incPushTime(start)

    loss = loss / dots.length.toLong
    println(s"avgPullTime=$avgPullTime gradTime=$avgCalTime " +
      s"avgMakeGradTime=$avgMakeGradTime avgPushTime=$avgPushTime loss=$loss")

    loss * batchSize
  }

  def dot(srcNodes: Array[Int], destNodes: Array[Int], negativeSamples: Array[Array[Int]],
          srcFeats: Int2ObjectOpenHashMap[Array[Float]], targetFeats: Int2ObjectOpenHashMap[Array[Float]], negative: Int): Array[Float] = {
    val dots: Array[Float] = new Array[Float]((1 + negative) * srcNodes.length)
    var docIndex = 0
    for (i <- srcNodes.indices) {
      val srcVec = srcFeats.get(srcNodes(i))
      // Get dot value for (src, dst)
      dots(docIndex) = arraysDot(srcVec, targetFeats.get(destNodes(i)))
      docIndex += 1

      // Get dot value for (src, negative sample)
      for (j <- 0 until negative) {
        dots(docIndex) = arraysDot(srcVec, targetFeats.get(negativeSamples(i)(j)))
        docIndex += 1
      }
    }
    dots
  }

  def arraysDot(x: Array[Float], y: Array[Float]): Float = {
    var dotValue = 0.0f
    x.indices.foreach(i => dotValue += x(i) * y(i))
    dotValue
  }

  def doGrad(dots: Array[Float], negative: Int, alpha: Float): Float = {
    var loss = 0.0f
    for (i <- dots.indices) {
      val prob = FastSigmoid.sigmoid(dots(i))
      if (i % (negative + 1) == 0) {
        dots(i) = alpha * (1 - prob)
        loss -= FastSigmoid.log(prob)
      } else {
        dots(i) = -alpha * FastSigmoid.sigmoid(dots(i))
        loss -= FastSigmoid.log(1 - prob)
      }
    }
    loss
  }

  def adjust(srcNodes: Array[Int], destNodes: Array[Int], negativeSamples: Array[Array[Int]],
             srcFeats: Int2ObjectOpenHashMap[Array[Float]], targetFeats: Int2ObjectOpenHashMap[Array[Float]],
             negative: Int, dots: Array[Float]) = {
    val inputUpdateCounter = new Int2IntOpenHashMap(srcFeats.size())
    val inputUpdates = new Int2ObjectOpenHashMap[Array[Float]](srcFeats.size())

    val outputUpdateCounter = new Int2IntOpenHashMap(targetFeats.size())
    val outputUpdates = new Int2ObjectOpenHashMap[Array[Float]](targetFeats.size())

    var docIndex = 0
    for (i <- srcNodes.indices) {
      // Src node grad
      val neule = new Array[Float](embeddingDim)

      // Accumulate dst node embedding to neule
      val dstEmbedding = targetFeats.get(destNodes(i))
      var g = dots(docIndex)
      axpy(neule, dstEmbedding, g)

      // Use src node embedding to update dst node embedding
      val srcEmbedding = srcFeats.get(srcNodes(i))
      mergeRaw(outputUpdateCounter, outputUpdates, destNodes(i), g, srcEmbedding)

      docIndex += 1

      // Use src node embedding to update negative sample node embedding; Accumulate negative sample node embedding to neule
      for (j <- 0 until negative) {
        val negSampleEmbedding = targetFeats.get(negativeSamples(i)(j))
        g = dots(docIndex)

        // Accumulate negative sample node embedding to neule
        axpy(neule, negSampleEmbedding, g)

        // Use src node embedding to update negative sample node embedding
        mergeRaw(outputUpdateCounter, outputUpdates, negativeSamples(i)(j), g, srcEmbedding)
        docIndex += 1
      }

      // Use accumulation to update src node embedding, grad = 1
      mergeRaw(inputUpdateCounter, inputUpdates, srcNodes(i), 1, neule)
    }

    var iter = inputUpdateCounter.int2IntEntrySet().fastIterator()
    while (iter.hasNext) {
      val entry = iter.next()
      div(inputUpdates.get(entry.getIntKey), entry.getIntValue.toFloat)
    }

    iter = outputUpdateCounter.int2IntEntrySet().fastIterator()
    while (iter.hasNext) {
      val entry = iter.next()
      div(outputUpdates.get(entry.getIntKey), entry.getIntValue.toFloat)
    }

    (inputUpdates, outputUpdates)
  }

  def axpy(y: Array[Float], x: Array[Float], a: Float) = {
    x.indices.foreach(i => y(i) += a * x(i))
  }

  def mergeRaw(inputUpdateCounter: Int2IntOpenHashMap, inputUpdates: Int2ObjectOpenHashMap[Array[Float]],
             nodeId: Int, g: Float, update: Array[Float]) = {
    var grads: Array[Float] = inputUpdates.get(nodeId)
    if (grads == null) {
      grads = new Array[Float](embeddingDim)
      inputUpdates.put(nodeId, grads)
      inputUpdateCounter.put(nodeId, 0)
    }

    //grads.iaxpy(update, g)
    axpy(grads, update, g)
    inputUpdateCounter.addTo(nodeId, 1)
  }

  def makeWord2vecEmbeddings(srcNodes: Array[Int], dstNodes: Array[Int], negativeSamples: Array[Array[Int]],
                             srcFeats: Int2ObjectOpenHashMap[Array[Float]], dstFeats: Int2ObjectOpenHashMap[Array[Float]], negative: Int): (Array[Float], Array[Float], Array[Float]) = {
    val srcEmbeddings = new ArrayBuffer[Float]()
    val dstEmbeddings = new ArrayBuffer[Float]()
    val negativeEmbeddings = new ArrayBuffer[Float]()
    for(i <- srcNodes.indices) {
      srcEmbeddings ++= srcFeats.get(srcNodes(i))
      dstEmbeddings ++= dstFeats.get(dstNodes(i))
      for(j <- 0 until negative) {
        negativeEmbeddings ++= dstFeats.get(negativeSamples(i)(j))
      }
    }
    (srcEmbeddings.toArray, dstEmbeddings.toArray, negativeEmbeddings.toArray)
  }

  def makeWord2VecGrad(srcEmbeddings: Array[Float], dstEmbeddings:Array[Float], negativeEmbeddings: Array[Float],
                       srcNodes: Array[Int], dstNodes: Array[Int], negativeSamples: Array[Array[Int]], srcFeatsSize: Int,
                       dstFeatsSize: Int, numNegSample: Int, learningRate: Float): (Int2ObjectOpenHashMap[Array[Float]], Int2ObjectOpenHashMap[Array[Float]]) = {
    val inputUpdateCounter = new Int2IntOpenHashMap(srcFeatsSize)
    val inputUpdates = new Int2ObjectOpenHashMap[Array[Float]](srcFeatsSize)

    val outputUpdateCounter = new Int2IntOpenHashMap(dstFeatsSize)
    val outputUpdates = new Int2ObjectOpenHashMap[Array[Float]](dstFeatsSize)
    for(i <- srcNodes.indices) {
      //update src
      val srcUpdateEmbedding = srcEmbeddings.slice(i*embeddingDim, (i+1)*embeddingDim)
      merge(inputUpdateCounter, inputUpdates, srcNodes(i), srcUpdateEmbedding, learningRate)
      //update dst
      val dstUpdateEmbedding = dstEmbeddings.slice(i*embeddingDim, (i+1)*embeddingDim)
      merge(outputUpdateCounter, outputUpdates, dstNodes(i), dstUpdateEmbedding, learningRate)
      //update negative example
      for(j <- 0 until numNegSample) {
        val negativeExampleUpdateEmbedding = negativeEmbeddings.slice(i*numNegSample*embeddingDim + j*embeddingDim, i*numNegSample*embeddingDim + (j+1)*embeddingDim)
        merge(outputUpdateCounter, outputUpdates, negativeSamples(i)(j), negativeExampleUpdateEmbedding, learningRate)
      }
    }

    var iter = inputUpdateCounter.int2IntEntrySet().fastIterator()
    while (iter.hasNext) {
      val entry = iter.next()
      div(inputUpdates.get(entry.getIntKey), entry.getIntValue.toFloat)
    }

    iter = outputUpdateCounter.int2IntEntrySet().fastIterator()
    while (iter.hasNext) {
      val entry = iter.next()
      div(outputUpdates.get(entry.getIntKey), entry.getIntValue.toFloat)
    }
    (inputUpdates, outputUpdates)
  }

  def merge(updateCounter: Int2IntOpenHashMap, updates: Int2ObjectOpenHashMap[Array[Float]],
            nodeId: Int, update: Array[Float], learningRate: Float): Int = {
    var grads: Array[Float] = updates.get(nodeId)
    if (grads == null) {
      grads = new Array[Float](embeddingDim)
      updates.put(nodeId, grads)
      updateCounter.put(nodeId, 0)
    }
    update.indices.foreach(i => grads(i) += -1.0f * learningRate * update(i))
    updateCounter.addTo(nodeId, 1)
  }

  def div(x: Array[Float], f: Float): Unit = {
    x.indices.foreach(i => x(i) = x(i) / f)
  }

  def logTime(msg: String): Unit = {
    val time = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date)
    println(s"[$time] $msg")
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


object Word2VecModel {

  def parseBatchData(sentences: Array[Array[Int]], windowSize: Int, negative: Int, maxIndex: Int, seed: Int = Random.nextInt): (Array[Int], Array[Int], Array[Array[Int]])= {
    val rand = new Random(seed)
    val srcNodes = new ArrayBuffer[Int]()
    val dstNodes = new ArrayBuffer[Int]()
    val negativeSamples = new ArrayBuffer[Array[Int]]()
    for (s <- sentences.indices) {
      val sen = sentences(s)
      for(srcIndex <- sen.indices) {

        val sampleRand = new Random(rand.nextInt())
        val sampleWords = new Array[Int](negative)
        var sampleIndex: Int = 0
        while (sampleIndex < negative) {
          val target = sampleRand.nextInt(maxIndex)
          if (target != sen(srcIndex)) {
            sampleWords(sampleIndex) = target
            sampleIndex += 1
          }
        }

        var dstIndex = Math.max(srcIndex - windowSize, 0)
        while (dstIndex < Math.min(srcIndex + windowSize + 1, sen.length)) {
          if(srcIndex != dstIndex) {
            srcNodes.append(sen(dstIndex))
            dstNodes.append(sen(srcIndex))
            negativeSamples.append(sampleWords)
          }
          dstIndex += 1
        }
      }
    }
    //val negativeSamples = negativeSample(srcNodes.toArray, dstNodes.toArray, negative, maxIndex, rand.nextInt())
    (srcNodes.toArray, dstNodes.toArray, negativeSamples.toArray)
  }

  def negativeSample(srcNodes: Array[Int], dstNodes: Array[Int], sampleNum: Int, maxIndex: Int, seed: Int): Array[Array[Int]] = {
    val rand = new Random(seed)
    val sampleWords = new Array[Array[Int]](srcNodes.length)
    var wordIndex: Int = 0

    for (i <- srcNodes.indices) {
      var sampleIndex: Int = 0
      sampleWords(wordIndex) = new Array[Int](sampleNum)
      while (sampleIndex < sampleNum) {
        val target = rand.nextInt(maxIndex)
        if (target != srcNodes(i) && target != dstNodes(i)) {
          sampleWords(wordIndex)(sampleIndex) = target
          sampleIndex += 1
        }
      }
      wordIndex += 1
    }
    sampleWords
  }

}
