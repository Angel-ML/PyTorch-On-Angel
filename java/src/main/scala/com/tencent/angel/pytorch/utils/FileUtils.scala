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
package com.tencent.angel.pytorch.utils

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext

import java.io.File
import java.text.SimpleDateFormat
import java.util.Calendar

object FileUtils {
  def getPtName(dir: String): String = {
    val path: File = new File(dir)
    val arrFile = path.listFiles.filter(_.isFile).filter(t => t.toString.endsWith(".pt"))
    arrFile(0).toString
  }

  def deletePath(input: String, sc: SparkContext): Unit = {
    val path = new Path(input)
    val fs = path.getFileSystem(sc.hadoopConfiguration)
    if (fs.exists(path) || fs.isDirectory(path)) {
      println(s"================delete exist path: ${path.toString}=======================")
      val flag = fs.delete(path, true)
      println(s"delete exit path $flag")
    }
  }

  def getFileName(dir: String, suffix: String): String = {
    val path: File = new File(dir)
    val arrFile = path.listFiles.filter(_.isFile).filter(t => t.toString.endsWith(suffix))
    var fileName: String = null
    if (suffix.equals(Constants.MODEL_FILE_SUFFIX)) {
      if (arrFile.nonEmpty) {
        fileName = arrFile(0).toString
        println("PyTorch model is exist, path is: " + fileName)
      } else {
        println("PyTorch model not exist, system will generate automatically!")
      }
    } else if (suffix.equals(Constants.CONFIG_FILE_SUFFIX)) {
      if (arrFile.isEmpty) {
        println("Error: Model yaml config file not exist, please upload!!!")
        arrFile(0).toString
        System.exit(-1)
      } else {
        fileName = arrFile(0).toString
        println("Model yaml config file is exist, path is: " + fileName)
      }
    } else if (suffix.equals(Constants.PYTHON_FILE_SUFFIX)) {
      if (arrFile.nonEmpty) {
        fileName = arrFile(0).toString
        println("Model python file uploaded by user, path is: " + fileName)
      } else {
        println("User did not upload python model file, system python model file will be used!")
      }
    } else {
      throw new Exception("Unsupported file suffix: " + suffix)
    }
    fileName
  }

  def parseDatePartitionName(path: String): String = {
    if (path.contains("$")) {
      val pos = path.indexOf("$")
      val start = if (path.contains("{")) 1 else 0
      val end = if (path.contains("}")) {
        path.indexOf("}")
      } else if (path.substring(pos).contains("/")) {
        pos + path.substring(pos).indexOf("/")
      } else {
        path.length
      }

      val base = path.substring(0, pos)
      val suffix = if (path.length == end || path.length == end + 1) "" else path.substring(pos + path.substring(pos).indexOf("/"))
      val partName = path.substring(pos + 1 + start, end).trim.replaceAll("\\s*", "")
      val format = new SimpleDateFormat("YYYYMMdd")
      val calendar = Calendar.getInstance()

      val part = if (partName.contains("-")) {
        val idx = partName.indexOf("-")
        val num = partName.substring(idx + 1, partName.length).toInt
        calendar.add(Calendar.DAY_OF_MONTH, -num)
        format.format(calendar.getTime)
      } else if (partName.contains("+")) {
        val idx = partName.indexOf("+")
        val num = partName.substring(idx + 1, partName.length).toInt
        calendar.add(Calendar.DAY_OF_MONTH, +num)
        format.format(calendar.getTime)
      } else {
        format.format(calendar.getTime)
      }

      println(s"parse path with date partition from: ${path} to ${base + part + suffix}")
      base + part + suffix
    } else {
      path
    }
  }
}