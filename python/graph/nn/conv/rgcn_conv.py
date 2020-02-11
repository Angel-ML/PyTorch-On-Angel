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
from utils import uniform
from utils import scatter_add, scatter_mean


class RGCNConv(torch.jit.ScriptModule):
    __constants__ = ['in_h', 'out_h', "n_relations", 'n_bases']

    def __init__(self, in_h, out_h, n_relations, n_bases):
        super(RGCNConv, self).__init__()

        self.in_h = in_h
        self.out_h = out_h
        self.n_relations = n_relations
        self.n_bases = n_bases

        self.basis = Parameter(torch.zeros(n_bases, in_h, out_h))
        self.att = Parameter(torch.zeros(n_relations, n_bases))
        self.root = Parameter(torch.zeros(in_h, out_h))
        self.bias = Parameter(torch.zeros(out_h))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.n_bases * self.in_h
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    @torch.jit.script_method
    def forward(self, x, edge_index, edge_type, edge_norm):
        # type: (Optional[Tensor], Tensor, Tensor, Optional[Tensor]) -> Tensor
        w = torch.matmul(self.att, self.basis.view(self.n_bases, -1))

        if x is None:
            w = w.view(-1, self.out_h)
            index = edge_type * self.in_h + edge_index[1]
            out = torch.index_select(w, 0, index)
        else:
            x_j = torch.index_select(x, 0, edge_index[1])
            w = w.view(self.n_relations, self.in_h, self.out_h)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        if edge_norm is not None:
            out = out * edge_norm.view(-1, 1)
        out = scatter_add(out, edge_index[0], dim=0)

        if x is None:
            out = out + self.root
        else:
            out = out + torch.matmul(x[0:out.size(0)], self.root)

        out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_h, self.out_h,
            self.n_relations)