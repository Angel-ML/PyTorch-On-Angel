# Tencent is pleased to support the open source community by making Angel available.
#
# Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/Apache-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
#

import torch


@torch.jit.script
def torch_float(value):
    # type: (float) -> Tensor
    a = torch.zeros(1)
    a[0] = float(value)
    return a


@torch.jit.script
def torch_int(value):
    # type: (int) -> Tensor
    a = torch.zeros(1, dtype=torch.int32)
    a[0] = int(value)
    return a


@torch.jit.script
def uniform(size, tensor):
    # type:(int, Optional[Tensor])
    size = torch_float(float(size))
    bound = 1.0 / torch.sqrt(size)
    bound = float(bound)
    if tensor is not None:
        torch.nn.init.uniform_(tensor, -bound, bound)


@torch.jit.script
def kaiming_uniform(tensor, fan, a):
    # type: (Optional[Tensor], float, float)
    if tensor is not None:
        bound = float(torch_float(6.0 / ((1.0 + a ** 2) * fan)))
        torch.nn.init.uniform_(tensor, -bound, bound)


@torch.jit.script
def glorot(tensor):
    # type: (Optional[Tensor])
    if tensor is not None:
        stdv = torch.sqrt(torch_float(6.0 / float((tensor.size(-2) + tensor.size(-1)))))
        k = stdv[0]
        torch.nn.init.uniform_(tensor, -k, k)


@torch.jit.script
def zeros(tensor):
    # type: (Optional[Tensor])
    if tensor is not None:
        tensor.fill_(0)


@torch.jit.script
def ones(tensor):
    # type: (Optional[Tensor])
    if tensor is not None:
        tensor.fill_(1)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
