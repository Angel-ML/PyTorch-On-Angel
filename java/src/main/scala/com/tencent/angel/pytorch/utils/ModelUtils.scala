package com.tencent.angel.pytorch.utils

import com.tencent.angel.model.output.format.MatrixFilesMeta
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext

import java.io.IOException

object ModelUtils {

  def recountPartition(psPartitionNum: Int, featEmbedPath: String): Int = {
    if (featEmbedPath.nonEmpty) {
      val basePath = new Path(featEmbedPath)
      val fs = basePath.getFileSystem(SparkContext.getOrCreate().hadoopConfiguration)
      var count = 0
      if (fs.exists(basePath) && fs.isDirectory(basePath)) {
        val files = fs.listFiles(basePath, true)//node Embedding files
        while (files.hasNext) {
          val metaPath = files.next().getPath
          if (fs.exists(metaPath) && metaPath.getName == "_meta") {
            println(s"metaPath: ${metaPath}")
            val meta = new MatrixFilesMeta
            val input = fs.open(metaPath)
            try {
              meta.read(input)
            } catch {
              case e: Throwable =>
                throw new IOException("Read meta failed ", e)
            } finally input.close()

            val partKeys = meta.getPartMetas.keySet().iterator()
            while(partKeys.hasNext) {
              val partId = partKeys.next().toInt
              if (partId > count) count = partId
            }
          }
        }
      }
      if (count != 0) count + 1 else psPartitionNum
    } else psPartitionNum
  }
}