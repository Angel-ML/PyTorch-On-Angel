package com.tencent.angel.pytorch.init

object InitUtils {
  def apply(initMethod: String, fin: Long = -1, fout: Long = -1, mean: Float = 0.0f, std: Float = 1.0f, gain: Float = 1.0f): InitMethod = {

    if (initMethod.startsWith("xavier"))
      assert(fin > 0 && fout > 0, s"fin: ${fin} and fout: ${fout} must be positive in xavierUniform or xavierNormal")
    if (initMethod.startsWith("kaiming"))
      assert(fin > 0, s"fin: ${fin} must be positive in kaimingUniform or kaimingNormal")

    initMethod match {
      case "randomUniform" => new RandomUniform(mean, std, fout.toInt)
      case "randomNormal" => new RandomNormal(mean, std)
      case "xavierUniform" => new XavierOrKaimingUniform(initMethod, fin, fout)
      case "kaimingUniform" => new XavierOrKaimingUniform(initMethod, fin)
      case "xavierNormal" => new XavierOrKaimingNormal(initMethod, fin, fout)
      case "kaimingNormal" => new XavierOrKaimingNormal(initMethod, fin)
      case "constant" => new Constant()
    }
  }
}
