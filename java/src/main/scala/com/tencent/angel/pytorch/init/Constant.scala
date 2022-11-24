package com.tencent.angel.pytorch.init


class Constant(constant: Float = 0.0f) extends InitMethod {

    override def getType(): Int = -1

    override def getFloats(): Array[Float] = Array(constant)

    override def getInts(): Array[Int] = Array(getType())
}
