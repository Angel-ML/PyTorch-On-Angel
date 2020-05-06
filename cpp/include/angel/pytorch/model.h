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
        EMBEDDINGS_MATS = 5,
        INVALID = -1,
    };


    class TorchModel {
    public:
        explicit TorchModel(const std::string &path) {
          module_ = torch::jit::load(path);
          module_.to(at::kCUDA);
        }

        void train() {
          module_.train();
        }

        void eval() {
          module_.eval();
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
            if (targets.defined())
              backward_inputs.emplace_back(targets.to(at::kCUDA));
            auto loss = module_.get_method("loss")(backward_inputs).toTensor();
            loss.backward();
            return loss.item().toFloat();
          } else if (outputs.isTuple()) {
            auto elements = outputs.toTuple()->elements();
            if (targets.defined())
              elements.emplace_back(targets.to(at::kCUDA));
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
          if (type == "EMBEDDINGS_MATS")
            return TorchModelType(5);
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

        std::vector<int> get_embeddings_size() {
          std::vector<int> sizes;
          auto eles_att = module_.get_attribute("embeddings_size").toIntList();
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

        std::string get_name() {
          return get_string("get_name");
        }

        int get_parameters_total_size();

        std::vector<at::Tensor> get_parameters();

        std::vector<at::Tensor> get_mats_parameters();

        void set_gcn_parameters(void *data_ptr, int size);

        void set_gcn_gradients(void *data_ptr, int size);

        void zero_grad();

        void set_parameter(const std::string& key, const torch::jit::IValue& value);

        void save_module(std::vector<torch::jit::IValue> parameters,
                         angel::TorchModelType type);

        void save(const std::string& path);

    private:
        torch::jit::script::Module module_;
    };
} // namespace angel


#endif //PYTORCH_TORCH_MODEL_H