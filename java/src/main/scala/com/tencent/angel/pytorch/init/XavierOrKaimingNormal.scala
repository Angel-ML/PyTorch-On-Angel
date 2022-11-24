package com.tencent.angel.pytorch.init

class XavierOrKaimingNormal(initMethod: String, fin: Long, fout: Long = -1, gain: Float = 1.0f) extends InitMethod {

  override def getType(): Int = 3

  override def getFloats(): Array[Float] = {
    if (initMethod == "xavierNormal")
      Array(gain * math.sqrt(2.0 / (fin + fout)).toFloat)
    else
      Array(gain * math.sqrt(2.0 / fin).toFloat)
  }

  override def getInts(): Array[Int] = Array(getType())
}
