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

import torch_scatter
from utils import scatter_add
from .num_nodes import maybe_num_nodes

@torch.jit.script
def softmax(src, index, num_nodes=None):
    # type: (Tensor, Tensor, Optional[int]) -> Tensor

    num_nodes = maybe_num_nodes(index, num_nodes)

    out, argmax = torch_scatter.scatter_max(src, index, dim=0, dim_size=num_nodes)
    out = src - out[index]
    out = out.exp() / (
        scatter_add(out.exp(), index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out