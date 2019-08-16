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
from .inits import torch_float, torch_int
import torch_scatter


@torch.jit.script
def maybe_dim_size(index, dim_size=None):
    # type: (Tensor, Optional[int]) -> int
    if dim_size is not None:
        return dim_size
    return int(index.max().item()) + 1 if index.numel() > 0 else 0

@torch.jit.script
def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    # type: (Tensor, Tensor, int, Optional[Tensor], Optional[int], int) -> Tuple[Tensor, Tensor, Tensor, int]
    # dim = src.dim() - 1
    # dim = range(src.dim())[dim]  # Get real dim value.
    dims = torch.jit.annotate(List[int], [])
    for i in range(src.dim()):
        dims.append(i)
    dim = dims[dim]
    # print(src.size())
    # print(index.size())

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = torch.jit.annotate(List[int], [])
        # index_size = []
        for i in range(src.dim()):
            index_size.append(1)
        index_size[dim] = src.size(dim)
        # print(index_size)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = torch.empty(out_size, dtype=src.dtype)
        out.fill_(fill_value)

    return src, out, index, dim

@torch.jit.script
def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    # type: (Tensor, Tensor, int, Optional[Tensor], Optional[int], int) -> Tensor
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)

@torch.jit.script
def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    # type: (Tensor, Tensor, int, Optional[Tensor], Optional[int], int) -> Tensor
    out = scatter_add(src, index, dim, out, dim_size, fill_value)
    count = scatter_add(torch.ones_like(src), index, dim, None, out.size(dim))
    return out / count.clamp(min=1)