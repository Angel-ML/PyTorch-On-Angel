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

    std::vector<at::Tensor> TorchModel::get_mats_parameters() {
        std::vector<at::Tensor> tensors;
        auto mats = module_.get_attribute("mats").toTensorList();
        for (size_t pos = 0; pos < mats.size(); pos++) {
            tensors.push_back(mats.get(pos).view({-1}));
        }
        return tensors;
    }

    void TorchModel::set_gcn_parameters(void *data_ptr, int size) {
        assert(size == get_parameters_total_size());
        // sub modules
        auto *ptr = reinterpret_cast<float *>(data_ptr);
        for (auto const &m : module_.get_modules()) {
            for (auto const &p : m.get_parameters()) {
                auto tensor = p.value().toTensor().to(at::kCPU);
                auto len = static_cast<size_t>(tensor.numel());
                memcpy(tensor.data_ptr(), reinterpret_cast<void*>(ptr), len * sizeof(float));
                ptr += len;
            }
        }
        // parameter from this module
        for (auto const &it: module_.get_parameters()) {
            auto tensor = it.value().toTensor().to(at::kCPU);
            auto len = static_cast<size_t>(tensor.numel());
            memcpy(tensor.data_ptr(), reinterpret_cast<void*>(ptr), len * sizeof(float));
            ptr += len;
        }
    }

    void TorchModel::set_gcn_gradients(void *data_ptr, int size) {
        assert(size == get_parameters_total_size());
        // clear grads
        memset(data_ptr, 0, size * sizeof(float));
        // sub modules
        auto *ptr = reinterpret_cast<float*>(data_ptr);
        for (auto const &m : module_.get_modules()) {
            for (auto const &p : m.get_parameters()) {
                auto tensor = p.value().toTensor();
                int64_t len = tensor.numel();
                if (tensor.grad().defined()) {
                    assert(len == tensor.grad().numel());
                    memcpy(ptr, tensor.grad().to(at::kCPU).data_ptr<float>(), len * sizeof(float));
                }
                ptr += len;
            }
        }
        // parameter from this module
        for (auto const &it: module_.get_parameters()) {
            auto tensor = it.value().toTensor();
            int64_t len = tensor.numel();
            if (tensor.grad().defined()) {
                assert(len == tensor.grad().numel());
                memcpy(ptr, tensor.grad().to(at::kCPU).data_ptr<float>(), len * sizeof(float));
            }
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

    void TorchModel::set_parameter(const std::string &key, const torch::jit::IValue &value) {
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

    void TorchModel::save(const std::string &path) {
        module_.save(path);
    }
} // namespace angel