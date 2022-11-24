package com.tencent.angel.pytorch.init

class RandomUniform(min: Float = -0.5f, max: Float = 0.5f, dim: Int = 1) extends InitMethod {

  override def getType(): Int = 0

  override def getFloats(): Array[Float] = Array(min, max)

  override def getInts(): Array[Int] = Array(dim, getType())
}
