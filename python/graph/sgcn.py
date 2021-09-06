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

import argparse

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from utils import scatter_mean


class SGCN(torch.jit.ScriptModule):

    def __init__(self, in_dim, out_dim):
        super(SGCN, self).__init__()
        self.weight = Parameter(torch.zeros(in_dim , out_dim))
        self.bias = Parameter(torch.zeros(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

    @torch.jit.script_method
    def forward_(self, x, first_edge_index, second_edge_index):
        embedding = self.embedding_(x, first_edge_index, second_edge_index)
        print(embedding.shape)
        out = torch.matmul(embedding, self.weight)
        out = out + self.bias
        return F.log_softmax(out, dim=1)

    @torch.jit.script_method
    def loss(self, y_pred, y_true):
        y_true = y_true.view(-1).to(torch.long)
        return F.nll_loss(y_pred, y_true)

    @torch.jit.script_method
    def predict_(self, x, first_edge_index, second_edge_index):
        output = self.forward_(x, first_edge_index, second_edge_index)
        return output.max(1)[1]

    @torch.jit.script_method
    def embedding_predict_(self, embedding):
        out = torch.matmul(embedding, self.weight)
        out = out + self.bias
        return F.log_softmax(out, dim=1)

    @torch.jit.script_method
    def embedding_(self, x, first_edge_index, second_edge_index):
        # first layer
        row, col = second_edge_index[0], second_edge_index[1]
        out = scatter_mean(x[col], row, dim=0)  # do not set dim_size
        #out = torch.cat([x[0:out.size(0)], out], dim=1)
        #out = torch.matmul(out, self.weight1)
        out = out + x[0:out.size(0)]
        out = out.div(2) 
        
        row, col = first_edge_index[0], first_edge_index[1]        
        neighbors = scatter_mean(out[col], row, dim=0)  # do not set dim_size
        #out = torch.cat([out[0:neighbors.size(0)], neighbors], dim=1)
        out = neighbors + x[0:neighbors.size(0)]
        out = out.div(2)
        print("embedding is here")
        print(out.shape)
        return out

    def acc(self, y_pred, y_true):
        y_true = y_true.view(-1).to(torch.long)
        return y_pred.max(1)[1].eq(y_true).sum().item() / y_pred.size(0)


FLAGS = None


def main():
    sage = SGCN(FLAGS.input_dim, FLAGS.output_dim)
    sage.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="input dimention of node features")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=-1,
        help="hidden dimension of graphsage convolution layer")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=-1,
        help="output dimension, the number of labels")
    parser.add_argument(
        "--output_file",
        type=str,
        default="graphsage.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
