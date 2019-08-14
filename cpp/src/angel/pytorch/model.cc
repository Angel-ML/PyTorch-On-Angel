//
// Created by leleyu on 2019-06-13.
//

#include <angel/pytorch/model.h>
#include <angel/commons.h>

namespace angel {

int TorchModel::get_parameters_total_size() {
  int size = 0;
  for (auto const &m: module_.get_modules())
    for (auto const &p: m.get_parameters())
      size += p.value().toTensor().numel();

  for (auto const &it: module_.get_parameters())
    size += it.value().toTensor().numel();

  return size;
}

std::vector<at::Tensor> TorchModel::get_parameters() {
  std::vector<at::Tensor> tensors;
  for (auto const& m: module_.get_modules())
    for (auto const &p: m.get_parameters())
      tensors.push_back(p.value().toTensor().view({-1}));

  for (auto const &it: module_.get_parameters())
    tensors.push_back(it.value().toTensor().view({-1}));

  return tensors;
}

void TorchModel::set_parameters(void *data_ptr, int size) {
  assert(size == get_parameters_total_size());
  // sub modules
  auto *ptr = reinterpret_cast<float *>(data_ptr);
  for (auto const &m : module_.get_modules()) {
    for (auto const &p : m.get_parameters()) {
      auto tensor = p.value().toTensor();
      size_t len = tensor.numel();
      memcpy(tensor.data_ptr(), reinterpret_cast<void*>(ptr), len * sizeof(float));
      ptr += len;
    }
  }
  // parameter from this module
  for (auto const &it: module_.get_parameters()) {
    auto tensor = it.value().toTensor();
    size_t len = tensor.numel();
    memcpy(tensor.data_ptr(), reinterpret_cast<void*>(ptr), len * sizeof(float));
    ptr += len;
  }
}

void TorchModel::set_grads(void *data_ptr, int size) {
  assert(size == get_parameters_total_size());
  // sub modules
  auto *ptr = reinterpret_cast<float*>(data_ptr);
  for (auto const &m : module_.get_modules()) {
    for (auto const &p : m.get_parameters()) {
      auto tensor = p.value().toTensor();
      size_t len = tensor.grad().numel();
      memcpy(ptr, tensor.grad().data<float>(), len * sizeof(float));
      ptr += len;
    }
  }

  // parameter from this module
  for (auto const &it: module_.get_parameters()) {
    auto tensor = it.value().toTensor();
    size_t len = tensor.grad().numel();
    memcpy(ptr, tensor.grad().data<float>(), len * sizeof(float));
    ptr += len;
  }
}

void TorchModel::zero_grad() {
  for (auto const &m : module_.get_modules()) {
    for (auto const &p : m.get_parameters()) {
      auto tensor = p.value().toTensor();
      if (tensor.grad().defined()) {
        tensor.grad().detach_();
        tensor.grad().zero_();
      }
    }
  }

  for (auto const &it : module_.get_parameters()) {
    auto tensor = it.value().toTensor();
    if (tensor.grad().defined()) {
      tensor.grad().detach_();
      tensor.grad().zero_();
    }
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


} // namespace angel