package com.tencent.angel.pytorch.examples

import com.tencent.angel.pytorch.model.{ParTorchModel, TorchParams}
import com.tencent.angel.pytorch.native.LibraryLoader
import com.tencent.angel.pytorch.optim.AsyncAdam
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.core.metric.AUC
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.{SparkConf, SparkContext}

object LocalExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "")
    val batchSize = params.getOrElse("batchSize", "100").toInt
    val torchModelPath = params.getOrElse("torchModelPath", "model.pt")
    val stepSize = params.getOrElse("stepSize", "0.1").toFloat
    val modelPath = params.getOrElse("modelPath", "")
    val modulePath = params.getOrElse("modulePath", "")

    val conf = new SparkConf()
    conf.setMaster("local[1]")
    conf.setAppName("local torch example")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    PSContext.getOrCreate(sc)
    val data = sc.textFile(input)

    val optim = new AsyncAdam(stepSize)
    val model = new ParTorchModel(optim, torchModelPath)
    model.init()

    for (epoch <- 1 to 10) {
      val epochStartTime = System.currentTimeMillis()
      val (lossSum, size) = data.mapPartitions {
        case iterator =>
          LibraryLoader.load
          iterator.sliding(batchSize, batchSize)
            .map(batch => (model.optimize(batch.toArray), batch.length))
      }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))


      val scores = data.mapPartitions {
        case iterator =>
          iterator.sliding(batchSize, batchSize)
            .map(batch => (model.predict(batch.toArray)))
            .flatMap(f => f._1.zip(f._2))
            .map(f => (f._1.toDouble, f._2.toDouble))
      }

      val auc = new AUC().calculate(scores)

      val epochTime = System.currentTimeMillis() - epochStartTime
      println(s"epoch=$epoch loss=${lossSum / size} auc=$auc time=${epochTime}ms")
    }

    if (modelPath.length > 0)
      model.save(modelPath)

    if (modulePath.length > 0)
      model.saveModule(modulePath, SparkHadoopUtil.get.newConfiguration(sc.getConf))

    PSContext.stop()
    sc.stop()
  }

}
