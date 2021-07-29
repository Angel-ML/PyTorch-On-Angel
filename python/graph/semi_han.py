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

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from nn.conv import HANConv
from utils import parse_feat


class HAN(torch.jit.ScriptModule):

    def __init__(self, in_dim, m, item_types, hidden, out_dim,
        task_type, input_embedding_dim, field_num, encode):
        super(HAN, self).__init__()
        # loss func for multi label classification
        self.loss_fn = torch.nn.BCELoss()
        self.task_type = task_type

        self.input_dim = in_dim
        self.embedding_dim = input_embedding_dim
        self.n_fields = field_num
        self.encode = encode

        # the transformed dense feature dimension for node
        in_dim = input_embedding_dim * field_num if input_embedding_dim > 0 else in_dim

        self.conv = HANConv(in_dim, m, item_types, hidden)

        self.weight = Parameter(torch.zeros(m, out_dim))
        self.bias = Parameter(torch.zeros(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

        if self.bias.dim() > 1:
            torch.nn.init.xavier_normal_(self.bias)

    @torch.jit.script_method
    def forward_(self, x, second_edge_index, second_edge_types,
        batch_ids=torch.Tensor([]), field_ids=torch.Tensor([])):
        embedding = self.embedding_(x, second_edge_index, second_edge_types, batch_ids, field_ids)
        out = torch.matmul(embedding, self.weight)
        out = out + self.bias
        if self.task_type == "classification":
            return F.log_softmax(out, dim=1)
        else:
            return F.sigmoid(out)

    @torch.jit.script_method
    def loss(self, y_pred, y_true):
        if self.task_type == "classification":
            y_true = y_true.view(-1).to(torch.long)
            return F.nll_loss(y_pred, y_true)
        else:
            u_true = y_true.reshape(y_pred.size())
            return self.loss_fn(y_pred, u_true)

    @torch.jit.script_method
    def predict_(self, x,  second_edge_index, second_edge_types,
        batch_ids=torch.Tensor([]), field_ids=torch.Tensor([])):
        output = self.forward_(x, second_edge_index, second_edge_types, batch_ids, field_ids)
        if self.task_type == "classification":
            return output.max(1)[1]
        else:
            return output

    @torch.jit.script_method
    def embedding_predict_(self, embedding):
        out = torch.matmul(embedding, self.weight)
        out = out + self.bias
        if self.task_type == "classification":
            return F.log_softmax(out, dim=1)
        else:
            return F.sigmoid(out)

    @torch.jit.script_method
    def embedding_(self, x, edge_index, edge_types,
        batch_ids=torch.Tensor([]), field_ids=torch.Tensor([])):
        if self.n_fields > 0:
            x = parse_feat(x, batch_ids, field_ids, self.n_fields, self.encode)
        return self.conv(x, edge_index, edge_types)


FLAGS = None


def main():
    han = HAN(FLAGS.input_dim, FLAGS.m, FLAGS.item_types,
              FLAGS.hidden_dim, FLAGS.output_dim, FLAGS.task_type,
              FLAGS.input_embedding_dim, FLAGS.input_field_num, FLAGS.encode)

    han.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m",
        type=int,
        default=-1,
        help="dimension of transform matrix")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="input dimension of node(user) features")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=-1,
        help="hidden dimension of nonlinear transformation at semantic-level aggregation")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=-1,
        help="output dimension, the number of labels")
    parser.add_argument(
        "--item_types",
        type=int,
        default=-1,
        help="the num of item_types")
    parser.add_argument(
        "--task_type",
        type=str,
        default="classification",
        help="classification or multi-label-classification")
    parser.add_argument(
        "--input_embedding_dim",
        type=int,
        default=-1,
        help="embedding dim of node features")
    parser.add_argument(
        "--input_field_num",
        type=int,
        default=-1,
        help="field num of node features")
    parser.add_argument(
        "--encode",
        type=str,
        default="dense",
        help="data encode, could be one-hot, multi-hot or dense")
    parser.add_argument(
        "--output_file",
        type=str,
        default="semi_han.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
