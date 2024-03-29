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
// Created by leleyu on 2019-06-13.
//

#include <angel/commons.h>
#include <angel/pytorch/model.h>

namespace angel {

int TorchModel::get_parameters_total_size() {
  int size = 0;
#ifdef LIBTORCH_VERSION_LATEST
  // auto iter and get all the parameters in the module struct and sub-module
  // struct
  for (auto const &p : module_.parameters())
    size += p.numel();
#else
  for (auto const &t : this->get_parameters())
    size += t.numel();
#endif

  return size;
}

std::vector<at::Tensor> TorchModel::get_parameters() {
  std::vector<at::Tensor> tensors;
#ifdef LIBTORCH_VERSION_LATEST
  // auto iter and get all the parameters in the module struct and sub-module
  // struct
  for (auto const &p : module_.parameters())
    tensors.push_back(p.view({-1}));
#else
  // this may not work for multi-nested modules
  // for (auto const &m: module_.get_modules())
  //     for (auto const &p: m.get_parameters())
  //       ...........
  // for (auto const &it: module_.get_parameters())
  //    ...........
  // get all the tensors by a iter
  geten_module_iters(module_, tensors);
#endif

  return tensors;
}

std::vector<at::Tensor> TorchModel::get_mats_parameters() {
  std::vector<at::Tensor> tensors;
  auto mats = module_.get_attribute("mats").toTensorList();
  for (size_t pos = 0; pos < mats.size(); pos++) {
    tensors.push_back(mats.get(pos).view({-1}));
  }
  return tensors;
}

void TorchModel::set_gcn_parameters(void *data_ptr, int size) {
#ifdef LIBTORCH_VERSION_LATEST
  assert(size == get_parameters_total_size());
  auto *ptr = reinterpret_cast<float *>(data_ptr);
  for (auto const &p : module_.parameters()) {
    auto len = static_cast<size_t>(p.numel());
    memcpy(p.data_ptr<float>(), reinterpret_cast<void *>(ptr),
           len * sizeof(float));
    ptr += len;
  }
#else
  auto tensors = get_parameters();

  int all_tensor_size = 0;
  for (auto const &t : tensors) {
    all_tensor_size += t.numel();
  }
  assert(all_tensor_size == size);

  auto *ptr = reinterpret_cast<float *>(data_ptr);
  for (auto const &t : tensors) {
    auto len = static_cast<size_t>(t.numel());
    memcpy(t.data_ptr<float>(), reinterpret_cast<void *>(ptr),
           len * sizeof(float));
    ptr += len;
  }
#endif
}

void TorchModel::set_gcn_gradients(void *data_ptr, int size) {
#ifdef LIBTORCH_VERSION_LATEST
  assert(size == get_parameters_total_size());
  // clear grads
  memset(data_ptr, 0, size * sizeof(float));
  auto *ptr = reinterpret_cast<float *>(data_ptr);
  for (auto const &p : module_.parameters()) {
    int64_t len = p.numel();
    if (p.grad().defined()) {
      assert(len == p.grad().numel());
      memcpy(ptr, p.grad().data_ptr<float>(), len * sizeof(float));
    }
    ptr += len;
  }
#else
  auto tensors = get_parameters();

  int all_tensor_size = 0;
  for (auto const &t : tensors) {
    all_tensor_size += t.numel();
  }
  assert(all_tensor_size == size);
  memset(data_ptr, 0, size * sizeof(float));

  auto *ptr = reinterpret_cast<float *>(data_ptr);
  for (auto const &t : tensors) {
    int64_t len = static_cast<size_t>(t.numel());
    if (t.grad().defined()) {
      assert(len == t.grad().numel());
      memcpy(ptr, t.grad().data_ptr<float>(), len * sizeof(float));
    }
    ptr += len;
  }
#endif
}

void TorchModel::zero_grad() {
#ifdef LIBTORCH_VERSION_LATEST
  for (auto const &p : module_.parameters()) {
    if (p.grad().defined()) {
      p.grad().detach_();
      p.grad().zero_();
    }
  }
#else
  for (auto const &t : this->get_parameters()) {
    if (t.grad().defined()) {
      t.grad().detach_();
      t.grad().zero_();
    }
  }
#endif
}

void TorchModel::set_parameter(const std::string &key,
                               const torch::jit::IValue &value) {
  if (value.isTensor()) {
    module_.get_parameter(key).set_data(value.toTensor());
  } else if (value.isTensorList()) {
    auto hosts = module_.get_attribute(key).toTensorList();
    auto values = value.toTensorList();
    for (size_t i = 0; i < hosts.size(); i++) {
      hosts.get(i).set_data(values.get(i));
    }
  } else {
    throw std::logic_error("set parameter should be tensor or tensor list");
  }
}

void TorchModel::save_module(std::vector<torch::jit::IValue> parameters,
                             angel::TorchModelType type) {
  using namespace angel;
  switch (type) {
  case angel::TorchModelType::BIAS_WEIGHT:
    module_.get_parameter("bias").set_data(parameters[0].toTensor());
    module_.get_parameter("weights").set_data(parameters[1].toTensor());
    break;
  case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING:
    module_.get_parameter("bias").set_data(parameters[0].toTensor());
    module_.get_parameter("weights").set_data(parameters[1].toTensor());
    module_.get_parameter("embedding").set_data(parameters[2].toTensor());
    break;
  case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS:
  case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD: {
    module_.get_parameter("bias").set_data(parameters[0].toTensor());
    module_.get_parameter("weights").set_data(parameters[1].toTensor());
    module_.get_parameter("embedding").set_data(parameters[2].toTensor());
    auto eles_para = parameters[3].toTensorList();
    auto eles_att = module_.get_attribute("mats").toTensorList();
    for (size_t pos = 0; pos < eles_para.size(); pos++) {
      eles_att.get(pos).set_data(eles_para.get(pos));
    }
    break;
  }
  default:
    break;
  }
  module_.save(get_name() + "-model.pt");
}

void TorchModel::save(const std::string &path) { module_.save(path); }
} // namespace angel