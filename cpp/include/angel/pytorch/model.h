//
// Created by leleyu on 2019-05-05.
//

#ifndef PYTORCH_TORCH_MODEL_H
#define PYTORCH_TORCH_MODEL_H

#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>
#include <jni.h>

namespace angel {


enum TorchModelType {
  BIAS_WEIGHT = 1,
  BIAS_WEIGHT_EMBEDDING = 2,
  BIAS_WEIGHT_EMBEDDING_MATS = 3,
  BIAS_WEIGHT_EMBEDDING_MATS_FIELD = 4,
  INVALID = -1,
};


class TorchModel {
 public:
  explicit TorchModel(const std::string &path) {
    module_ = torch::jit::load(path);
  }

  c10::IValue forward(std::vector<torch::jit::IValue> inputs) {
    return module_.get_method("forward_")(std::move(inputs));
  }

  c10::IValue predict(std::vector<torch::jit::IValue> inputs) {
    return module_.get_method("predict_")(std::move(inputs));
  }

  c10::IValue exec_method(const std::string &method, std::vector<torch::jit::IValue> inputs) {
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
    for (auto const &f: list)
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
    return TorchModelType(-1);
  }

  std::string get_type_string() {
    return get_string("get_type");
  }

  std::vector<int> get_mats_size() {
    std::vector<int> sizes;
    auto eles_att = module_.get_attribute("mats").toTensorList();
    for (size_t pos = 0; pos < eles_att.size(); pos++) {
      auto ele = eles_att.get(pos);
      for (auto &f: ele.sizes()) {
        sizes.push_back(static_cast<int>(f));
      }
    }
    return sizes;
  }

  int64_t get_input_dim() {
    return module_.get_attribute("input_dim").toInt();
  }

  int64_t get_num_fields() {
    return module_.get_attribute("n_fields").toInt();
  }

  int64_t get_embedding_dim() {
    return module_.get_attribute("embedding_dim").toInt();
  }

  std::string get_name() {
    return get_string("get_name");
  }

  int get_parameters_total_size();

  std::vector<at::Tensor> get_parameters();

  void set_parameters(void *data_ptr, int size);

  void set_grads(void *data_ptr, int size);

  void zero_grad();

  void save_module(std::vector<torch::jit::IValue> parameters,
                   angel::TorchModelType type);

 private:
//  std::shared_ptr<torch::jit::script::Module> module_;
  torch::jit::script::Module module_;
};
} // namespace angel


#endif //PYTORCH_TORCH_MODEL_H
