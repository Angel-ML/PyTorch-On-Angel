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
from utils import scatter_add


class HANConv(torch.jit.ScriptModule):

    def __init__(self, in_dim, m, item_types, hidden):
        super(HANConv, self).__init__()

        self.m = m
        self.M = Parameter(torch.zeros(in_dim, m))
        self.alpha = Parameter(torch.zeros(item_types, m * 2))
        self.q = Parameter(torch.zeros(1, hidden))

        self.weight = Parameter(torch.zeros(m, hidden))
        self.bias = Parameter(torch.zeros(hidden))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.xavier_normal_(self.alpha)
        torch.nn.init.xavier_normal_(self.q)
        if self.bias.dim() > 1:
            torch.nn.init.xavier_normal_(self.bias)

    @torch.jit.script_method
    def forward(self, x, edge_index, edge_types):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        x = torch.mm(x, self.M)

        # node-level aggregation
        out, second_input, new_index = self.node_level(x, edge_index, edge_types)

        # semantic-level attention
        out = self.semantic_level(out, second_input, new_index, edge_types)

        return out

    @torch.jit.script_method
    def node_level(self, x, edge_index, edge_types):
        row, col = edge_index[0], edge_index[1]
        exp = torch.relu(torch.mul(torch.cat((x[row], x[col]), dim=1),
                                   self.alpha[edge_types])).sum(1).exp()
        new_index = row * self.item_types + edge_types
        softmax = scatter_add(exp, new_index)[new_index]
        att_weight1 = torch.div(exp, softmax)
        out = scatter_add(x[col] * att_weight1.reshape(x[col].size()[0], 1), new_index, 0)
        out = out[torch._unique(new_index)[0]]
        out = torch.relu(out)
        # out = F.normalize(out, p=2.0, dim=-1)

        # nonlinear transformation
        second_input = torch.tanh(torch.matmul(out, self.weight) + self.bias)

        return out, second_input, new_index

    @torch.jit.script_method
    def semantic_level(self, out, input, edge_index, edge_types):
        temp = torch._unique(edge_index)[0]
        new_row_index = torch.div(temp, self.item_types)
        new_type_index = torch.remainder(temp, self.item_types)
        count = scatter_add(torch.ones_like(edge_index), edge_types) + 1e-16
        a = scatter_add(input, new_type_index, 0)
        wq = torch.mul(torch.div(a, count.reshape(a.size()[0], 1)), self.q).sum(1).exp()
        softmax = wq.sum()
        beta = torch.div(wq, softmax)
        out = scatter_add(out * beta[new_type_index].reshape(out.size()[0], 1), new_row_index, 0)

        out = F.normalize(out, p=2.0, dim=-1)

        return out