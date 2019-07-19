package com.tencent.angel.pytorch.native

object LibraryLoader {

  val libFiles = Array(
//    "gomp",
//    "c10",
//    "caffe2_detectron_ops",
//    "foxi",
//    "torch",
//    "caffe2",
//    "caffe2_module_test_dynamic",
//    "foxi_dummy",
    "torch_angel")


  def loadFunc(): Boolean = {

    for (i <- libFiles.indices)
      System.loadLibrary(libFiles(i))

    return true
  }


  lazy val load = loadFunc()
}
