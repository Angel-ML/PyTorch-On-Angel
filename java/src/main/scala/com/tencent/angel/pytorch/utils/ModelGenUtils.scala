package com.tencent.angel.pytorch.utils

import com.tencent.angel.pytorch.utils.YamlParserUtils.{GNNParameter, RecommendationParameter}
import org.apache.spark.deploy.PythonRunner

import java.util
import scala.collection.JavaConverters._
import scala.collection.mutable

object ModelGenUtils {

  def genPyTorchModel(model: mutable.Map[String, Object], genModelType: String): String = {
    val model_name = model(Constants.MODEL_NAME).toString
    var param = new StringBuilder()
    model.foreach { ele =>
      param.append("--")
      param.append(ele._1)
      param.append("=")
      ele._2 match {
        case ints: util.ArrayList[Int] =>
          param.append(ints.toArray().mkString(" "))
        case _ =>
          param.append(ele._2)
      }
      param.append(" ")
    }
    param = param.init
    println("PyTorch Model gen parameter is: " + param.toString())
    val modelPythonFile = FileUtils.getFileName(Constants.FILE_BASE_PATH, Constants.PYTHON_FILE_SUFFIX)
    val pythonFile  = if (modelPythonFile == null) {
      println("python model path: " + Constants.PYTHON_ENV_MODEL_BASE_PATH + genModelType + "/" + model_name + Constants.PYTHON_FILE_SUFFIX)
      Constants.PYTHON_ENV_MODEL_BASE_PATH + genModelType + "/" + model_name + Constants.PYTHON_FILE_SUFFIX
    } else {
      modelPythonFile
    }
    val pyFiles = Constants.PYTHON_ENV_MODEL_BASE_PATH + genModelType + "," +
      Constants.PYTHON_ENV_MODEL_BASE_PATH + genModelType.split("/")(0)
    PythonRunner.main(Array(pythonFile, pyFiles, param.toString()))
    println("PyTorch Model gen finished!")
    FileUtils.getFileName(Constants.FILE_BASE_PATH, Constants.MODEL_FILE_SUFFIX)
  }

  def getGNNTorchModelPathAndConfig(graphModelType: String, taskType: String): (String, GNNParameter) = {
    val torchModelFile = FileUtils.getFileName(Constants.FILE_BASE_PATH, Constants.MODEL_FILE_SUFFIX)
    val yamlConfigFile = FileUtils.getFileName(Constants.FILE_BASE_PATH, Constants.CONFIG_FILE_SUFFIX)
    val config = YamlParserUtils.parseGNN(yamlConfigFile, graphModelType)
    val torchModelPath = if (torchModelFile == null) {
      genPyTorchModel(config.model.asScala, Constants.GEN_MODEL_TYPE_GRAPH + "/" + graphModelType + "/" + taskType)
    } else {
      torchModelFile
    }
    (torchModelPath, config)
  }

  def getRecommTorchModelPathAndConfig(): (String, RecommendationParameter) = {
    val torchModelFile = FileUtils.getFileName(Constants.FILE_BASE_PATH, Constants.MODEL_FILE_SUFFIX)
    val yamlConfigFile = FileUtils.getFileName(Constants.FILE_BASE_PATH, Constants.CONFIG_FILE_SUFFIX)
    val config = YamlParserUtils.parseRecommendation(yamlConfigFile)
    val torchModelPath = if (torchModelFile == null) {
      genPyTorchModel(config.model.asScala, Constants.GEN_MODEL_TYPE_RECOMMENDATION)
    } else {
      torchModelFile
    }
    (torchModelPath, config)
  }

}
