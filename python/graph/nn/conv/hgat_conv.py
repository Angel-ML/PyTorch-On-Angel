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
#!/usr/bin/env python

from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from utils import scatter_add, softmax


class HGATConv(torch.jit.ScriptModule):
    __constants__ = ['heads', 'out_channels', 'negative_slope', 'dropout']

    def __init__(self,
        in_channels_u,
        in_channels_i,
        out_channels,
        heads=2,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
        act=False):
        super(HGATConv, self).__init__()
        self.in_channels_u = in_channels_u
        self.in_channels_i = in_channels_i
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act = act

        self.weight_u = Parameter(torch.Tensor(in_channels_u,
                                               heads * out_channels))
        self.weight_i = Parameter(torch.Tensor(in_channels_i,
                                               heads * out_channels))

        self.weight_u1 = Parameter(torch.Tensor(in_channels_u,
                                                heads * out_channels))

        self.att_u = Parameter(torch.Tensor(heads, out_channels, 1))

        self.att_i = Parameter(torch.Tensor(heads, out_channels, 1))

        self.att = Parameter(torch.Tensor(heads, 2 * out_channels, 1))

        self.alpha = Parameter(torch.zeros(heads * out_channels, 1))

        self.bias_u = Parameter(torch.zeros(heads * out_channels, 1))
        self.bias_i = Parameter(torch.zeros(heads * out_channels, 1))
        self.bias_u1 = Parameter(torch.zeros(heads * out_channels, 1))

        self.batchnorm = torch.nn.BatchNorm1d(heads * out_channels)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_u)
        torch.nn.init.xavier_uniform_(self.weight_i)
        torch.nn.init.xavier_uniform_(self.weight_u1)
        torch.nn.init.xavier_uniform_(self.att_u)
        torch.nn.init.xavier_uniform_(self.att_i)
        torch.nn.init.xavier_uniform_(self.att)

        if self.alpha.dim() > 1:
            torch.nn.init.xavier_uniform_(self.alpha)
        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

        if self.bias_u.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias_u)
        if self.bias_i.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias_i)
        if self.bias_u1.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias_u1)

    @torch.jit.script_method
    def forward(self, u, i, edge_index):
        # type: (Tensor, Tensor, Tensor) -> Tensor

        u_origin = torch.mm(u, self.weight_u1) + self.bias_u1.view(-1)
        u_ = (torch.mm(u, self.weight_u) + self.bias_u.view(-1))\
            .view(-1, self.heads, self.out_channels)
        i_ = (torch.mm(i, self.weight_i) + self.bias_i.view(-1))\
            .view(-1, self.heads, self.out_channels)

        out = self.propagate(edge_index, u_, i_, num_nodes=u_.size(0))
        # for euler GNN
        u_origin = (u_origin[0: out.size(0)])\
            .view(-1, self.heads * self.out_channels)
        out = out + u_origin

        out = self.batchnorm(out)

        if self.act:
            out = self.parametric_relu(out)

        # out size: (num_nodes, (heads * out_channels))
        return out

    @torch.jit.script_method
    def propagate(self, edge_index, u, i, num_nodes):
        # type: (Tensor, Tensor, Tensor, int) -> Tensor
        row = edge_index[0]
        out = self.message(edge_index, u, i, num_nodes)

        # (U, heads, out_dim)
        out = scatter_add(out, row, dim=0)

        # (U, heads * out_dim)
        out = out.view(-1, self.heads * self.out_channels)

        # out size: num_nodes * (heads * out_channels)
        return out

    @torch.jit.script_method
    def message(self, edge_index, x_u, x_i, num_nodes):
        # type: (Tensor, Tensor, Tensor, int) -> Tensor

        row, col = edge_index[0], edge_index[1]
        u = torch.index_select(x_u, 0, row)
        i = torch.index_select(x_i, 0, col)
        feat = torch.transpose(torch.cat([u, i], dim=2), 0, 1)
        attn_edge = torch.bmm(feat, self.att)
        attn_edge = torch.transpose(attn_edge, 0, 1)

        attn_edge = F.leaky_relu(attn_edge, self.negative_slope)
        attn_score = softmax(attn_edge, edge_index[0], None)
        attn_score = F.dropout(attn_score, p=self.dropout, training=self.training)
        x_i = torch.index_select(x_i, 0, col)

        out = torch.mul(attn_score, x_i).view(-1, self.heads, self.out_channels)

        # return size: (num_edges, heads, out_channels)
        return out

    @torch.jit.script_method
    def update(self, aggr_out):
        # type: (Tensor) -> Tensor
        return aggr_out + self.bias

    @torch.jit.script_method
    def parametric_relu(self, _x):
        pos = F.relu(_x)
        neg = self.alpha.view(-1) * (_x - abs(_x)) * 0.5

        return pos + neg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)