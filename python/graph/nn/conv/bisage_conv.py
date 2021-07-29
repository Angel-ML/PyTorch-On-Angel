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
from utils import scatter_mean


class BiSAGEConv(torch.jit.ScriptModule):
    def __init__(self, in_dim, neigh_dim, out_dim):
        super(BiSAGEConv, self).__init__()
        # bipartite graph, two types of nodes, namely u,i. Edges: (u, i)

        self.neigh_weight = Parameter(
            torch.zeros(neigh_dim, in_dim))  # transform matrix
        self.weight = Parameter(
            torch.zeros(in_dim * 2, out_dim))  # weight matrix
        self.bias = Parameter(torch.zeros(out_dim))
        self.activation = torch.nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.neigh_weight)
        torch.nn.init.xavier_uniform_(self.weight)

        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, x, neigh, edge_index):
        row, col = edge_index[0], edge_index[1]
        # do not set dim_size
        neighborhood = scatter_mean(neigh[col], row, dim=0)
        neighborhood = torch.matmul(neighborhood, self.neigh_weight)
        out = torch.cat([x[0:neighborhood.size(0)], neighborhood], dim=1)
        out = torch.matmul(out, self.weight)
        out = out + self.bias

        out = self.activation(out)
        # out = F.normalize(out, p=2.0, dim=-1)
        return out