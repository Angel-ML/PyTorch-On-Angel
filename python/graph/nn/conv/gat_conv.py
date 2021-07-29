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
import torch.nn.functional as F
from torch.nn import Parameter
from utils import remove_self_loops, add_self_loops, softmax
from utils import scatter_add


class GATConv(torch.jit.ScriptModule):
    __constants__ = ['heads', 'out_channels', 'negative_slope', 'dropout']

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.2,
                 bias=True):
        super(GATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))

        self.weight1 = Parameter(torch.Tensor(out_channels,
                                              heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.xavier_uniform_(self.weight1)
        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

    @torch.jit.script_method
    def forward(self, x, edge_index):
        # type: (Tensor, Tensor) -> Tensor
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    @torch.jit.script_method
    def propagate(self, edge_index, x, num_nodes):
        # type: (Tensor, Tensor, int) -> Tensor
        x_i = torch.index_select(x, 0, edge_index[0])
        x_j = torch.index_select(x, 0, edge_index[1])
        edge_index_i = edge_index[0]
        out = self.message(edge_index_i, x_i, x_j, num_nodes)

        out = scatter_add(out, edge_index_i, dim=0).sum(dim=1)
        out = self.update(torch.matmul(out, self.weight1))
        out = torch.softmax(out, dim=0)
        out = F.normalize(out, p=2.0, dim=-1)

        # out size: num_nodes * heads * out_channels
        # TODO: support multi-heads condition
        return out.squeeze(1)

    @torch.jit.script_method
    def message(self, edge_index_i, x_i, x_j, num_nodes):
        # type: (Tensor, Tensor, Tensor, int) -> Tensor

        # Compute the attention coefficients
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # return size: num_edges * heads * out_channels
        return x_j * alpha.view(-1, self.heads, 1)

    @torch.jit.script_method
    def update(self, aggr_out):
        # type: (Tensor) -> Tensor
        return aggr_out + self.bias

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
