package com.tencent.angel.pytorch.init

class RandomNormal(mean: Float = 0.0f, stdDev: Float = 1.0f) extends InitMethod {

  override def getType(): Int = 1

  override def getFloats(): Array[Float] = Array(mean, stdDev)

  override def getInts(): Array[Int] = Array(1, getType())
}
