/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
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
