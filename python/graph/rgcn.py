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

import argparse

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from nn.conv import RGCNConv


class RGCN(torch.jit.ScriptModule):

    def __init__(self, input_dim, hidden_dim, n_relations, n_bases, n_class):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, n_relations, n_bases)
        self.conv2 = RGCNConv(hidden_dim, n_class, n_relations, n_bases)

    @torch.jit.script_method
    def forward_(self, x, edge_index, edge_type, edge_norm):
        # type: (Optional[Tensor], Tensor, Tensor, Optional[Tensor]) -> Tensor
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        return F.log_softmax(x, dim=1)

    @torch.jit.script_method
    def loss(self, y_pred, y_true):
        y_true = y_true.view(-1).to(torch.long)
        return F.nll_loss(y_pred, y_true)


class RelationGCN(torch.jit.ScriptModule):

    def __init__(self, input_dim, hidden_dim, n_relations, n_bases, n_class):
        super(RelationGCN, self).__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, n_relations, n_bases)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, n_relations, n_bases)
        self.weight = Parameter(torch.zeros(hidden_dim, n_class))
        self.bias = Parameter(torch.zeros(n_class))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

    @torch.jit.script_method
    def forward_(self, x, first_edge_index, second_edge_index, first_edge_type, second_edge_type):
        # type: (Optional[Tensor], Tensor, Tensor, Tensor, Tensor) -> Tensor
        x = self.embedding_(x, first_edge_index, second_edge_index, first_edge_type, second_edge_type)
        x = torch.matmul(x, self.weight)
        x = x + self.bias
        return F.log_softmax(x, dim=1)

    @torch.jit.script_method
    def loss(self, y_pred, y_true):
        y_true = y_true.view(-1).to(torch.long)
        return F.nll_loss(y_pred, y_true)

    @torch.jit.script_method
    def predict_(self, x, first_edge_index, second_edge_index, first_edge_type, second_edge_type):
        # type: (Optional[Tensor], Tensor, Tensor, Tensor, Tensor) -> Tensor
        output = self.forward_(x, first_edge_index, second_edge_index, first_edge_type, second_edge_type)
        return output.max(1)[1]

    @torch.jit.script_method
    def embedding_(self, x, first_edge_index, second_edge_index, first_edge_type, second_edge_type):
        # type: (Optional[Tensor], Tensor, Tensor, Tensor, Tensor) -> Tensor
        x = F.relu(self.conv1(x, second_edge_index, second_edge_type, None))
        x = self.conv2(x, first_edge_index, first_edge_type, None)
        return x


FLAGS = None


def main():
    rgcn = RelationGCN(FLAGS.input_dim, FLAGS.hidden_dim, FLAGS.n_relations, FLAGS.n_bases, FLAGS.n_class)
    rgcn.save(FLAGS.output_file)#创建模型并保存


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="input dimension of node features")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="hidden dimension of rgcn convolution layer")
    parser.add_argument(
        "--n_class",
        type=int,
        default=2,
        help="the number of classes")
    parser.add_argument(
        "--output_file",
        type=str,
        default="rgcn.pt",
        help="output file name")
    parser.add_argument(
        "--n_relations",
        type=int,
        default=1,
        help="the number types of relations for edges")
    parser.add_argument(
        "--n_bases",
        type=int,
        default=30,
        help="the number of bases in rgcn model")
    FLAGS, unparsed = parser.parse_known_args()
    main()