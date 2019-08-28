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

from torch import Tensor
from typing import List

import math

class AttentionNet(torch.jit.ScriptModule):

    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, fc_dims=[]):
        super(AttentionNet, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.mats = []

        if input_dim > 0 and embedding_dim > 0 and n_fields > 0 and fc_dims:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1))
            self.weights = torch.nn.Parameter(torch.zeros(input_dim, 1))
            self.embedding = torch.nn.Parameter(torch.zeros(input_dim, embedding_dim))
            torch.nn.init.xavier_normal_(self.bias)
            torch.nn.init.xavier_normal_(self.weights)
            # wq, wk, wv, w1, w1
            for i in range(5):
                w = torch.zeros(embedding_dim, embedding_dim)
                torch.nn.init.xavier_normal_(w)
                self.mats.append(w)

            dim = n_fields * embedding_dim  # m * d
            for (index, fc_dim) in enumerate(fc_dims):
                w = torch.nn.Parameter(torch.zeros(dim, fc_dim))
                b = torch.nn.Parameter(torch.randn(1, 1))
                torch.nn.init.xavier_normal_(w)
                self.mats.append(w)
                self.mats.append(b)
                dim = fc_dim

            self.input_dim = torch.jit.Attribute(self.input_dim, int)
            self.n_fields = torch.jit.Attribute(self.n_fields, int)
            self.embedding_dim = torch.jit.Attribute(self.embedding_dim, int)
            self.mats = torch.jit.Attribute(self.mats, List[Tensor])


    @torch.jit.script_method
    def first_order(self, batch_size, index, values, bias, weights):
        # type: (int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        srcs = weights.view(1, -1).mul(values.view(1, -1)).view(-1)
        output = torch.zeros(batch_size, dtype=torch.float32)
        output.scatter_add_(0, index, srcs)
        first = output + bias
        return first

    @torch.jit.script_method
    def attention(self, batch_size, k, embedding, mats):
        # type: (int, int, Tensor, List[Tensor]) -> Tensor
        wq, wk, wv, w1, w2 = mats
        b = batch_size
        embedding = embedding.view(b, -1, k)

        Q = torch.relu(torch.matmul(embedding, wq))
        K = torch.relu(torch.matmul(embedding, wk))
        V = torch.relu(torch.matmul(embedding, wv))

        sqrk_k = torch.zeros(1, dtype=torch.float32)
        sqrk_k[0] = k
        sqrk_k = torch.sqrt(sqrk_k)

        QK_t = torch.matmul(Q, K.transpose(1, 2)).div(sqrk_k)
        V = torch.softmax(QK_t, dim=2).matmul(V) + embedding
        E = torch.relu(torch.matmul(V, w1))
        E = torch.matmul(E, w2) + V
        return E

    @torch.jit.script_method
    def higher_order(self, batch_size, embedding, mats):
        # type: (int, Tensor, List[Tensor]) -> Tensor
        k = embedding.size(1)
        e = self.attention(batch_size, k, embedding, mats[0:5])
        e = e.view(batch_size, -1)

        mats = mats[5:]
        for i in range(int(len(mats) / 2)):
            e = torch.relu(e.matmul(mats[i*2]) + mats[i*2 + 1])

        return e.view(-1)

    @torch.jit.script_method
    def forward_(self, batch_size, index, values, bias, weights, embedding, mats):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor]) -> Tensor
        first = self.first_order(batch_size, index, values, bias, weights)
        higher = self.higher_order(batch_size, embedding, mats)
        return torch.sigmoid(first + higher)

    @torch.jit.script_method
    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        batch_first = F.embedding(feats, self.weights)
        batch_embedding = F.embedding(feats, self.embedding)
        return self.forward_(batch_size, index, values,
                             self.bias, batch_first, batch_embedding, self.mats)

    @torch.jit.script_method
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.script_method
    def get_type(self):
        return "BIAS_WEIGHT_EMBEDDING_MATS"

    @torch.jit.script_method
    def get_name(self):
        return "AttentionNet"


FLAGS = None


def main():
    attention = AttentionNet(FLAGS.input_dim, FLAGS.n_fields, FLAGS.embedding_dim, FLAGS.fc_dims)
    attention.save('attention_net.pt')


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
