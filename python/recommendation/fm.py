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


class FactorizationMachine(torch.nn.Module):

    def __init__(self, input_dim=-1, embedding_dim=-1):
        super(FactorizationMachine, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # local model do not need real input_dim to init params, so set fake_input_dim to
        # speed up to produce local pt file.
        fake_input_dim = 10
        if input_dim > 0 and embedding_dim > 0:
            self.bias = torch.randn(1, 1, dtype=torch.float32)
            self.weights = torch.randn(fake_input_dim, 1)
            self.embedding = torch.randn(fake_input_dim, embedding_dim)
            self.bias = torch.nn.Parameter(self.bias, requires_grad=True)
            self.weights = torch.nn.Parameter(self.weights, requires_grad=True)
            self.embedding = torch.nn.Parameter(self.embedding, requires_grad=True)
            torch.nn.init.xavier_uniform_(self.weights)
            torch.nn.init.xavier_uniform_(self.embedding)

    def first_order(self, batch_size, index, values, bias, weights):
        # type: (int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        size = batch_size
        srcs = weights.view(1, -1).mul(values.view(1, -1)).view(-1)
        output = torch.zeros(size, dtype=torch.float32, device="cuda")
        output.scatter_add_(0, index, srcs)
        first = output + bias
        return first

    def second_order(self, batch_size, index, values, embeddings):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        k = embeddings.size(1)
        b = batch_size

        # t1: [k, n]
        t1 = embeddings.mul(values.view(-1, 1)).transpose_(0, 1)
        # t1: [k, b]
        t1_ = torch.zeros(k, b, dtype=torch.float32)

        for i in range(k):
            t1_[i].scatter_add_(0, index, t1[i])

        # t1: [k, b]
        t1 = t1_.pow(2)

        # t2: [k, n]
        t2 = embeddings.pow(2).mul(values.pow(2).view(-1, 1)).transpose_(0, 1)
        # t2: [k, b]
        t2_ = torch.zeros(k, b, dtype=torch.float32)
        for i in range(k):
            t2_[i].scatter_add_(0, index, t2[i])

        # t2: [k, b]
        t2 = t2_

        second = t1.sub(t2).transpose_(0, 1).sum(1).mul(0.5)
        return second

    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
        first = self.first_order(batch_size, index, values, bias, weights)
        second = self.second_order(batch_size, index, values, embeddings)
        return torch.sigmoid(first + second)

    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        batch_first = F.embedding(feats, self.weights)
        batch_second = F.embedding(feats, self.embedding)
        return self.forward_(batch_size, index, feats, values, self.bias, batch_first, batch_second)

    @torch.jit.export
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.export
    def get_type(self):
        return "BIAS_WEIGHT_EMBEDDING"

    @torch.jit.export
    def get_name(self):
        return "FactorizationMachine"


FLAGS = None


def main():
    fm = FactorizationMachine(FLAGS.input_dim, FLAGS.embedding_dim)
    fm_script_module = torch.jit.script(fm)
    fm_script_module.save("fm.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="data input dim."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=-1,
        help="embedding dim."
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
