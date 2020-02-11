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
from nn.conv import SAGEConv3


class DGITwoOrder(torch.jit.ScriptModule):

    def __init__(self, n_in, n_h1, n_h2):
        super(DGITwoOrder, self).__init__()
        self.gcn1 = SAGEConv3(n_in, n_h1)
        self.gcn2 = SAGEConv3(n_h1, n_h2)
        self.weight = Parameter(torch.Tensor(n_h2, n_h2))
        self.prelu1 = Parameter(torch.Tensor(n_h1).fill_(0.25))
        self.prelu2 = Parameter(torch.Tensor(n_h2).fill_(0.25))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    @torch.jit.script_method
    def forward_(self, pos_x, neg_x, first_edge_index, second_edge_index):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        pos_x = F.prelu(self.gcn1(pos_x, second_edge_index), self.prelu1)
        pos_z = F.prelu(self.gcn2(pos_x, first_edge_index), self.prelu2)
        neg_x = F.prelu(self.gcn1(neg_x, second_edge_index), self.prelu1)
        neg_z = F.prelu(self.gcn2(neg_x, first_edge_index), self.prelu2)
        summary = torch.sigmoid(torch.mean(pos_z, dim=0))
        return pos_z, neg_z, summary

    @torch.jit.script_method
    def loss(self, pos_z, neg_z, summary):
        r"""Evaluates the loss using the difference between
        the mutual information of positive and negative patches
        with global summary."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()

        return pos_loss + neg_loss

    @torch.jit.script_method
    def discriminate(self, z, summary, sigmoid=True):
        # type: (Tensor, Tensor, bool) -> Tensor
        r"""Evaluates the probablity score of the patch given the global summary.
        Does not apply the sigmoid function to the discrimination output when sigmoid is set to False.
        """
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    @torch.jit.script_method
    def embedding_(self, x, first_edge_index, second_edge_index):
        r"""Generates learned representation of the input nodes."""
        x = F.prelu(self.gcn1(x, second_edge_index), self.prelu1)
        x = F.prelu(self.gcn2(x, first_edge_index), self.prelu2)
        return x


FLAGS = None


def main():
    dgi2 = DGITwoOrder(FLAGS.input_dim, FLAGS.hidden_dim, FLAGS.output_dim)
    dgi2.save(FLAGS.output_file)


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
        help="hidden dimension of dgi convolution layer")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=32,
        help="output dimension of dgi")
    parser.add_argument(
        "--output_file",
        type=str,
        default="dgi2.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
