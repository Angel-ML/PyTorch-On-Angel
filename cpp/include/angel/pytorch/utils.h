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
