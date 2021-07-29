/*
 * Tencent is pleased to support the open source community by making Angel
 * available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights
 * reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 *
 */
//
// Created by leleyu on 2019-05-05.
//

#ifndef PYTORCH_TORCH_MODEL_H
#define PYTORCH_TORCH_MODEL_H

#include <iostream>

#include <jni.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace angel {

enum TorchModelType {
  BIAS_WEIGHT = 1,
  BIAS_WEIGHT_EMBEDDING = 2,
  BIAS_WEIGHT_EMBEDDING_MATS = 3,
  BIAS_WEIGHT_EMBEDDING_MATS_FIELD = 4,
  EMBEDDINGS_MATS = 5,
  EMBEDDINGS_MATS_FIELD = 6,
  INVALID = -1,
};

#ifdef LIBTORCH_VERSION_LATEST
using OrModule = torch::jit::script::Module;
using IValue = c10::IValue;
using parameter_list = torch::jit::parameter_list;
using module_list = torch::jit::module_list;
using ModulePtr = c10::intrusive_ptr<c10::ivalue::Object>;

struct WapperModule : public OrModule {
  explicit WapperModule() {}
  WapperModule(ModulePtr objPtr) : OrModule(std::move(objPtr)) {}
  ~WapperModule() {}

  IValue get_attribute(const std::string &name) const {
    try {
      return this->attr(name);
    } catch (...) {
      throw std::logic_error("No such attribute named as " + name);
    }
  }

  parameter_list get_parameters() const { return this->parameters(); }

  torch::autograd::Variable get_parameter(const std::string &name) const {
    for (auto const &p : this->named_parameters()) {
      if (name == p.name) {
        return torch::autograd::as_variable_ref(p.value);
      }
    }
  }

  module_list get_modules() const { return this->modules(); }
};
#endif

class TorchModel {
public:
  explicit TorchModel(const std::string &path) {
#ifdef LIBTORCH_VERSION_LATEST
    OrModule m_ = torch::jit::load(path);
    module_ = angel::WapperModule(std::move(m_._ivalue()));
#else
    module_ = torch::jit::load(path);
#endif
  }

  void train() { module_.train(); }

  void eval() { module_.eval(); }

  c10::IValue importance(std::vector<torch::jit::IValue> inputs) {
    return module_.get_method("importance_")(std::move(inputs));
  }

  c10::IValue forward(std::vector<torch::jit::IValue> inputs) {
    return module_.get_method("forward_")(std::move(inputs));
  }

  c10::IValue predict(std::vector<torch::jit::IValue> inputs) {
    return module_.get_method("predict_")(std::move(inputs));
  }

  c10::IValue exec_method(const std::string &method,
                          std::vector<torch::jit::IValue> inputs) {
    return module_.get_method(method)(std::move(inputs));
  }

  at::Tensor serving_forward(std::vector<torch::jit::IValue> inputs) {
    return module_.get_method("forward")(std::move(inputs)).toTensor();
  }

  float backward(std::vector<torch::jit::IValue> inputs, at::Tensor targets) {
    auto outputs = forward(std::move(inputs));
    if (outputs.isTensor()) {
      std::vector<torch::jit::IValue> backward_inputs;
      backward_inputs.emplace_back(outputs.toTensor());
      if (targets.defined())
        backward_inputs.emplace_back(targets);
      auto loss = module_.get_method("loss")(backward_inputs).toTensor();
      loss.backward();
      return loss.item().toFloat();
    } else if (outputs.isTuple()) {
      auto elements = outputs.toTuple()->elements();
      if (targets.defined())
        elements.emplace_back(targets);
      auto loss = module_.get_method("loss")(elements).toTensor();
      loss.backward();
      return loss.item().toFloat();
    } else {
      throw std::logic_error("The output of forward should be tensor or tuple");
    }
  }

  float recommend_backward(std::vector<torch::jit::IValue> inputs) {
    auto loss = forward(std::move(inputs)).toTensor();
    loss.backward();
    return loss.item().toFloat();
  }

  std::string get_string(const std::string &method) {
    std::vector<torch::jit::IValue> inputs;
    return module_.get_method(method)(inputs).toString()->string();
  }

  int64_t get_int64(const std::string &method) {
    std::vector<torch::jit::IValue> inputs;
    return module_.get_method(method)(inputs).toInt();
  }

  std::vector<int> get_vector(const std::string &method) {
    std::vector<torch::jit::IValue> inputs;
    auto list = module_.get_method(method)(inputs).toIntList();
    std::vector<int> sizes;
    for (auto const &f : list)
      sizes.push_back(static_cast<int>(f));
    return sizes;
  }

  TorchModelType get_type() {
    std::string type = get_string("get_type");
    if (type == "BIAS_WEIGHT")
      return TorchModelType(1);
    if (type == "BIAS_WEIGHT_EMBEDDING")
      return TorchModelType(2);
    if (type == "BIAS_WEIGHT_EMBEDDING_MATS")
      return TorchModelType(3);
    if (type == "BIAS_WEIGHT_EMBEDDING_MATS_FIELD")
      return TorchModelType(4);
    if (type == "EMBEDDINGS_MATS")
      return TorchModelType(5);
    if (type == "EMBEDDINGS_MATS_FIELD")
      return TorchModelType(6);
    return TorchModelType(-1);
  }

  std::string get_type_string() { return get_string("get_type"); }

  std::vector<int> get_mats_size() {
    std::vector<int> sizes;
    auto eles_att = module_.get_attribute("mats").toTensorList();
    for (size_t pos = 0; pos < eles_att.size(); pos++) {
      auto ele = eles_att.get(pos);
      for (auto &f : ele.sizes()) {
        sizes.push_back(static_cast<int>(f));
      }
    }
    return sizes;
  }

  int64_t get_input_dim() { return module_.get_attribute("input_dim").toInt(); }

  int64_t get_user_input_dim() {
    return module_.get_attribute("user_input_dim").toInt();
  }

  int64_t get_item_input_dim() {
    return module_.get_attribute("item_input_dim").toInt();
  }

  int64_t get_num_fields() { return module_.get_attribute("n_fields").toInt(); }

  int64_t get_user_num_fields() {
    return module_.get_attribute("user_field_num").toInt();
  }

  int64_t get_item_num_fields() {
    return module_.get_attribute("item_field_num").toInt();
  }

  int64_t get_embedding_dim() {
    return module_.get_attribute("embedding_dim").toInt();
  }

  int64_t get_user_embedding_dim() {
    return module_.get_attribute("user_embedding_dim").toInt();
  }

  int64_t get_item_embedding_dim() {
    return module_.get_attribute("item_embedding_dim").toInt();
  }

  std::vector<int> get_embeddings_size() {
    std::vector<int> sizes;
    auto eles_att = module_.get_attribute("embeddings_size").toIntList();
    for (size_t pos = 0; pos < eles_att.size(); pos++) {
      auto ele = eles_att.get(pos);
      sizes.push_back(static_cast<int>(ele));
    }
    return sizes;
  }

  std::vector<int> get_dense_col_nums() {
    std::vector<int> sizes;
    auto eles_att = module_.get_attribute("dense_col_nums").toIntList();
    for (size_t pos = 0; pos < eles_att.size(); pos++) {
      auto ele = eles_att.get(pos);
      sizes.push_back(static_cast<int>(ele));
    }
    return sizes;
  }

  std::vector<int> get_sparse_col_nums() {
    std::vector<int> sizes;
    auto eles_att = module_.get_attribute("sparse_col_nums").toIntList();
    for (size_t pos = 0; pos < eles_att.size(); pos++) {
      auto ele = eles_att.get(pos);
      sizes.push_back(static_cast<int>(ele));
    }
    return sizes;
  }

  std::vector<int64_t> get_inputs_size() {
    std::vector<int64_t> sizes;
    auto eles_att = module_.get_attribute("inputs_size").toIntList();
    for (size_t pos = 0; pos < eles_att.size(); pos++) {
      auto ele = eles_att.get(pos);
      sizes.push_back(static_cast<int64_t>(ele));
    }
    return sizes;
  }

  std::string get_name() { return get_string("get_name"); }

#ifndef LIBTORCH_VERSION_LATEST
  // iter for module and sub-module struct and get all parameter tensors
  void geten_module_iters(torch::jit::script::Module m_,
                          std::vector<at::Tensor> &tensors) {
    for (auto const &p : m_.get_parameters())
      tensors.push_back(p.value().toTensor());

    for (auto const &m : m_.get_slots())
      if (m.is_module())
        geten_module_iters(m.to_module(), tensors);
  }
#endif
  int get_parameters_total_size();

  std::vector<at::Tensor> get_parameters();

  std::vector<at::Tensor> get_mats_parameters();

  void set_gcn_parameters(void *data_ptr, int size);

  void set_gcn_gradients(void *data_ptr, int size);

  void zero_grad();

  void set_parameter(const std::string &key, const torch::jit::IValue &value);

  void save_module(std::vector<torch::jit::IValue> parameters,
                   angel::TorchModelType type);

  void save(const std::string &path);

private:
#ifndef LIBTORCH_VERSION_LATEST
  torch::jit::script::Module module_;
#else
  angel::WapperModule module_;
#endif
};
} // namespace angel

#endif // PYTORCH_TORCH_MODEL_H