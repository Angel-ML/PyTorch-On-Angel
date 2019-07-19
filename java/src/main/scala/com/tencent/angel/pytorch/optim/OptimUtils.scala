package com.tencent.angel.pytorch.optim

object OptimUtils {

  def apply(optim: String, stepSize: Double): AsyncOptim = {
    optim.toLowerCase match {
      case "sgd" => new AsyncSGD(stepSize)
      case "adam" => new AsyncAdam(stepSize)
      case "adagrad" => new AsyncAdagrad(stepSize)
      case "momentum" => new AsyncMomentum(stepSize)
    }
  }


}
