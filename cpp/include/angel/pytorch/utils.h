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
//
// Created by leleyu on 2019-05-17.
//

#ifndef TORCH_ON_ANGEL_UTILS_H
#define TORCH_ON_ANGEL_UTILS_H

#include <angel/commons.h>
#include <vector>
#include <unordered_map>

#include <torch/torch.h>

#include <jni.h>

namespace angel {

std::vector<at::Tensor> make_mat_tensors(void *data_ptr, void *size_ptr, int n_data, int n_size);

void set_mat_grads(JNIEnv *env, const std::vector<at::Tensor> &tensors, jfloatArray jmats,
                   void *size_ptr, int n_size);

void add_input(JNIEnv *env,
               std::vector<torch::jit::IValue> *inputs,
               std::vector<std::pair<std::string, void*>> *ptrs,
               jobject params,
               const std::vector<int64_t> &sizes,
               const torch::TensorOptions &option,
               const std::string &key);

void add_input(JNIEnv *env,
               std::vector<torch::jit::IValue> *inputs,
               std::vector<std::pair<std::string, void*>> *ptrs,
               jarray array,
               const std::vector<int64_t> &sizes,
               const torch::TensorOptions &option,
               const std::string &key);

void add_input(JNIEnv *env,
               std::vector<torch::jit::IValue> *inputs,
               std::vector<std::pair<std::string, void*>> *ptrs,
               jobject params,
               const torch::TensorOptions &option,
               const std::string &data_key,
               const std::string &size_key);

void add_input(JNIEnv *env,
               std::vector<torch::jit::IValue> *inputs,
               std::vector<std::pair<std::string, void*>> *ptrs,
               jobject params,
               const torch::TensorOptions &option,
               const std::string &key);

void add_input(JNIEnv *env,
               std::vector<std::pair<std::string, at::Tensor>> *params,
               std::vector<std::pair<std::string, void*>> *ptrs,
               jarray value,
               const torch::TensorOptions &option,
               const std::string &key);

void release_array(JNIEnv *env,
                   const std::vector<std::pair<std::string, void*>> &ptrs,
                   jobject params);

void set_grads(JNIEnv *env,
               std::vector<torch::jit::IValue> &inputs,
               std::vector<std::pair<std::string, void*>> &ptrs,
               jobject params);
}
#endif //TORCH_ON_ANGEL_UTILS_H
