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
# !/usr/bin/env python

import torch
from torch.nn import Parameter

from utils import add_remaining_self_loops
from utils import scatter_add
from utils import glorot, zeros

from typing import Tuple


class GCNConv(torch.jit.ScriptModule):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = 'add'

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        # zeros(self.bias)

    @torch.jit.script_method
    def norm(self, edge_index, num_nodes):
        # type: (Tensor, int) -> Tuple[Tensor, Tensor]
        edge_weight = torch.ones((edge_index.size(1),),
                                 device=edge_index.device)

        fill_value = 1
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    @torch.jit.script_method
    def forward(self, x, edge_index):
        # type: (Tensor, Tensor) -> Tensor
        x = torch.matmul(x, self.weight)
        edge_index, norm = self.norm(edge_index, x.size(0))

        return self.propagate(edge_index, x=x, norm=norm)

    @torch.jit.script_method
    def propagate(self, edge_index, x, norm):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        x_j = torch.index_select(x, 0, edge_index[1])
        out = self.message(x_j, norm)
        out = scatter_add(out, edge_index[0], 0, None, dim_size=x.size(0))
        out = self.update(out)
        return out

    @torch.jit.script_method
    def message(self, x_j, norm):
        # type: (Tensor, Tensor) -> Tensor
        return norm.view(-1, 1) * x_j

    @torch.jit.script_method
    def update(self, aggr_out):
        # type: (Tensor) -> Tensor
        return aggr_out + self.bias

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCNConv2(torch.jit.ScriptModule):
    '''
    GCNConv2 is designed for distributed mini-batch training.
    '''

    def __init__(self,
                 in_channels,
                 out_channels):
        super(GCNConv2, self).__init__()

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        # zeros(self.bias)

    @torch.jit.script_method
    def norm(self, edge_index, num_nodes):
        # type: (Tensor, int) -> Tuple[Tensor, Tensor]
        edge_weight = torch.ones((edge_index.size(1),),
                                 device=edge_index.device)

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        mask = deg_inv_sqrt == float('inf')
        if mask is not None:
            # deg_inv_sqrt[mask] = 0
            deg_inv_sqrt.masked_fill_(mask, 0)

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    @torch.jit.script_method
    def forward(self, x, edge_index):
        # type: (Tensor, Tensor) -> Tensor
        x = torch.matmul(x, self.weight)
        edge_index, norm = self.norm(edge_index, x.size(0))

        return self.propagate(edge_index, x=x, norm=norm)

    @torch.jit.script_method
    def propagate(self, edge_index, x, norm):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        x_j = torch.index_select(x, 0, edge_index[1])
        out = self.message(x_j, norm)
        out = scatter_add(out, edge_index[0], 0, None)  # do not set dim_size, out.size() = edge_index[1].max() + 1
        out = self.update(out)
        return out

    @torch.jit.script_method
    def message(self, x_j, norm):
        # type: (Tensor, Tensor) -> Tensor
        return norm.view(-1, 1) * x_j

    @torch.jit.script_method
    def update(self, aggr_out):
        # type: (Tensor) -> Tensor
        return aggr_out + self.bias

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)