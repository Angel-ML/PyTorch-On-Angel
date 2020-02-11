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

from torch import Tensor
from typing import List


class DeepAndWide(torch.nn.Module):

    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, fc_dims=[], encode="onehot"):
        super(DeepAndWide, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.encode = encode
        self.mats = []

        # local model do not need real input_dim to init params, so set fake_dim to
        # speed up to produce local pt file.
        fake_input_dim = 10
        if input_dim > 0 and embedding_dim > 0 and n_fields > 0 and fc_dims:
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.weights = torch.nn.Parameter(torch.zeros(fake_input_dim, 1))
            self.embedding = torch.nn.Parameter(torch.zeros(fake_input_dim, embedding_dim))
            torch.nn.init.xavier_uniform_(self.weights)
            torch.nn.init.xavier_uniform_(self.embedding)
            dim = n_fields * embedding_dim
            for (index, fc_dim) in enumerate(fc_dims):
                self.mats.append(torch.nn.Parameter(torch.randn(dim, fc_dim)))
                self.mats.append(torch.nn.Parameter(torch.zeros(1, 1)))
                torch.nn.init.kaiming_uniform_(self.mats[index * 2], mode='fan_in', nonlinearity='relu')
                dim = fc_dim

    def first_order(self, batch_size, index, values, bias, weights):
        # type: (int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        size = batch_size
        srcs = weights.view(1, -1).mul(values.view(1, -1)).view(-1)
        output = torch.zeros(size, dtype=torch.float32)
        output.scatter_add_(0, index, srcs)
        first = output + bias
        return first

    def higher_order(self, batch_size, index, embeddings, mats, fields):
        # type: (int, Tensor, Tensor, List[Tensor], Tensor) -> Tensor
        if self.encode == "onehot":
            b = batch_size
            output = embeddings.view(b, -1)
        else:
            k = embeddings.size(1)
            b = batch_size
            f = self.n_fields
            t_index = [index, fields]
            e_transpose = embeddings.view(-1, k).transpose(0, 1)
            count = torch.ones(embeddings.size(0))
            hs = []
            for i in range(k):
                h = torch.zeros(b, f)
                c = torch.zeros(b, f)
                h.index_put_(t_index, e_transpose[i], True)
                c.index_put_(t_index, count, True)
                h = h / c.clamp(min=1)
                hs.append(h.view(-1, 1))
            emb_cat = torch.cat(hs, dim=1)
            output = emb_cat.view(batch_size, -1)

        for i in range(int(len(mats) / 2)):
            output = torch.relu(output.matmul(mats[i * 2]) + mats[i * 2 + 1])

        return output.view(-1)

    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings, mats, fields=torch.Tensor([])):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor], Tensor) -> Tensor

        first = self.first_order(batch_size, index, values, bias, weights)
        higher = self.higher_order(batch_size, index, embeddings, mats, fields)

        return torch.sigmoid(first + higher)

    def forward(self, batch_size, index, feats, values, fields=torch.Tensor([])):
        # type: (int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        batch_first = F.embedding(feats, self.weights)
        batch_second = F.embedding(feats, self.embedding)
        return self.forward_(batch_size, index, feats, values,
                             self.bias, batch_first, batch_second, self.mats, fields)

    @torch.jit.export
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.export
    def get_type(self):
        if self.encode == "onehot":
            return "BIAS_WEIGHT_EMBEDDING_MATS"
        else:
            return "BIAS_WEIGHT_EMBEDDING_MATS_FIELD"

    @torch.jit.export
    def get_name(self):
        return "DeepAndWide"


FLAGS = None


def main():
    deepandwide = DeepAndWide(FLAGS.input_dim, FLAGS.n_fields, FLAGS.embedding_dim, FLAGS.fc_dims, FLAGS.encode)
    deepandwide_script_module = torch.jit.script(deepandwide)
    deepandwide_script_module.save("deepandwide.pt")


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
        "--n_fields",
        type=int,
        default=-1,
        help="data num fields."
    )
    parser.add_argument(
        "--encode",
        type=str,
        default="onehot",
        help="data encode."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=-1,
        help="embedding dim."
    )
    parser.add_argument(
        "--fc_dims",
        nargs="+",
        type=int,
        default=-1,
        help="fc layers dim list."
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()