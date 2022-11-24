package com.tencent.angel.pytorch.init

class XavierOrKaimingUniform(initMethod: String, fin: Long, fout: Long = -1, gain: Float = 1.0f) extends InitMethod {

  override def getType(): Int = 2

  override def getFloats(): Array[Float] = {
    if (initMethod == "xavierUniform")
      Array((math.sqrt(3.0f) * gain * math.sqrt(2.0f / (fin + fout))).toFloat)
    else
      Array((math.sqrt(3.0f) * gain * math.sqrt(2.0f / fin)).toFloat)
  }

  override def getInts(): Array[Int] = Array(getType())
}
