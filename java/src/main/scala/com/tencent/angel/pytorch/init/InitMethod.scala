package com.tencent.angel.pytorch.init

abstract class InitMethod extends Serializable {

  def getType(): Int

  def getFloats(): Array[Float]

  def getInts(): Array[Int]
}
