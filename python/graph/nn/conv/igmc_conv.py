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


from __future__ import print_function

import torch
from torch.nn import Parameter
from utils import glorot
from utils import scatter_add


class IGMCConv(torch.jit.ScriptModule):
    def __init__(self, out_dim, n_bases=30, edge_types=2):
        super(IGMCConv, self).__init__()
        # bipartite graph, two types of nodes, namely u,i. Edges: (u, i)

        self.weight = Parameter(
            torch.zeros(out_dim * 2, out_dim))  # weight matrix
        self.bias = Parameter(torch.zeros(out_dim))

        self.out_h = out_dim
        self.edge_types = edge_types
        self.n_bases = n_bases
        self.basis = Parameter(torch.zeros(n_bases, out_dim, out_dim))
        self.att = Parameter(torch.zeros(edge_types, n_bases))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)

        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

        glorot(self.basis)
        glorot(self.att)

    @torch.jit.script_method
    def forward(self, u, i, edge_index, edge_type):
        w = torch.matmul(self.att, self.basis.view(self.n_bases, -1))
        w = w.view(self.edge_types, self.out_h, self.out_h)
        row, col = edge_index[0], edge_index[1]

        # u-i, i-u
        u_embedding = self.embedding(w, u, i, row, col, edge_type)
        i_embedding = self.embedding(w, i, u, col, row, edge_type)

        return u_embedding, i_embedding

    @torch.jit.script_method
    def embedding(self, w, x, neigh, row, col, e_type):
        neighborhood = torch.index_select(neigh, 0, col)
        e_type = e_type.view(-1).to(torch.long)
        iw = torch.index_select(w, 0, e_type)

        neighborhood = torch.bmm(neighborhood.unsqueeze(1), iw).squeeze(-2)
        neighborhood = scatter_add(neighborhood, row, dim=0)

        out = neighborhood + x[0:neighborhood.size(0)]
        out = torch.tanh(out)

        # out = F.normalize(out, p=2.0, dim=-1)
        return out
