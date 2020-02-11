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
package com.tencent.angel.pytorch.model

object TorchModelType extends Enumeration {
  type TorchModelType = Value
  val BIAS_WEIGHT = Value("BIAS_WEIGHT")
  val EMBEDDINGS_MATS = Value("EMBEDDINGS_MATS")
  val BIAS_WEIGHT_EMBEDDING = Value("BIAS_WEIGHT_EMBEDDING")
  val BIAS_WEIGHT_EMBEDDING_MATS = Value("BIAS_WEIGHT_EMBEDDING_MATS")
  val BIAS_WEIGHT_EMBEDDING_MATS_FIELD = Value("BIAS_WEIGHT_EMBEDDING_MATS_FIELD")
}
