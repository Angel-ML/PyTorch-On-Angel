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


class AttentionNetMultiHead(torch.jit.ScriptModule):

    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, num_multi_head=-1,
                 top_k=-1, num_attention_layers=-1, fc_dims=[]):
        super(AttentionNetMultiHead, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.num_multi_head = num_multi_head
        self.top_k = top_k
        self.num_attention_layers = num_attention_layers
        self.mats = []

        if input_dim > 0 and embedding_dim > 0 and n_fields > 0 and num_multi_head > 0 \
                and top_k > 0 and num_attention_layers > 0 and fc_dims:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1))
            self.weights = torch.nn.Parameter(torch.zeros(input_dim, 1))
            self.embedding = torch.nn.Parameter(torch.zeros(input_dim, embedding_dim))
            torch.nn.init.xavier_normal_(self.weights)
            torch.nn.init.xavier_normal_(self.embedding)

            # wq,wk,wv, w1, w2 with multi-head
            for i in range(5 * num_attention_layers):
                w = torch.zeros(num_multi_head, embedding_dim, embedding_dim)
                torch.nn.init.xavier_normal_(w)
                self.mats.append(w.view(num_multi_head, embedding_dim*embedding_dim))

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
            self.num_multi_head = torch.jit.Attribute(self.num_multi_head, int)
            self.top_k = torch.jit.Attribute(self.top_k, int)
            self.num_attention_layers = torch.jit.Attribute(self.num_attention_layers, int)
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
    def attention(self, batch_size, k, embedding, num_multi_head, top_k, n_fields, mats):
        # type: (int, int, Tensor, int, int, int, List[Tensor]) -> Tensor
        assert top_k <= n_fields
        wq, wk, wv, w1, w2 = mats
        b = batch_size
        embedding = embedding.view(b, -1, k)
        embedding_ = embedding.unsqueeze(1).repeat(1, num_multi_head, 1, 1)
        Q = torch.relu(torch.matmul(embedding_, wq.view(num_multi_head, self.embedding_dim, self.embedding_dim)))
        K = torch.relu(torch.matmul(embedding_, wk.view(num_multi_head, self.embedding_dim, self.embedding_dim)))
        V = torch.relu(torch.matmul(embedding_, wv.view(num_multi_head, self.embedding_dim, self.embedding_dim)))

        sqrk_k = torch.zeros(1, dtype=torch.float32)
        sqrk_k[0] = k
        sqrk_k = torch.sqrt(sqrk_k.div(num_multi_head))

        QK_t = torch.matmul(Q, K.transpose(2, 3)).div(sqrk_k)
        QK_t = QK_t.view(-1, n_fields)
        topk = torch.topk(QK_t, top_k)
        topk_indices = topk.indices
        size_i, size_j = topk_indices.size()
        rows = torch.arange(size_i).to(torch.long).repeat_interleave(size_j).view(1, -1)
        QK_t.fill_(float('-inf'))
        QK_t.index_put_((rows, topk_indices.view(-1)), topk.values.view(-1))
        QK_t = QK_t.view(batch_size, num_multi_head, n_fields, n_fields)
        V = torch.softmax(QK_t, dim=3).matmul(V) + embedding_
        E = torch.relu(torch.matmul(V, w1.view(num_multi_head, self.embedding_dim, self.embedding_dim)))
        E = torch.matmul(E, w2.view(num_multi_head, self.embedding_dim, self.embedding_dim)) + V
        E = E.sum(1)
        return E

    @torch.jit.script_method
    def higher_order(self, batch_size, embedding, num_multi_head, top_k, n_fields, num_attention_layers, mats):
        # type: (int, Tensor, int, int, int, int, List[Tensor]) -> Tensor
        k = embedding.size(1)
        e = embedding
        for i in range(num_attention_layers):
            weights = mats[num_attention_layers * i: num_attention_layers * i + 5]
            e = self.attention(batch_size, k, e, num_multi_head, top_k, n_fields, weights)
        e = e.view(batch_size, -1)

        mats = mats[num_attention_layers * 5:]
        for i in range(int(len(mats) / 2)):
            e = torch.relu(e.matmul(mats[i * 2]) + mats[i * 2 + 1])

        return e.view(-1)

    @torch.jit.script_method
    def forward_(self, batch_size, index, values, bias, weights, embedding, mats):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor]) -> Tensor
        first = self.first_order(batch_size, index, values, bias, weights)
        higher = self.higher_order(batch_size, embedding, self.num_multi_head, self.top_k,
                                   self.n_fields, self.num_attention_layers, mats)
        return torch.sigmoid(first + higher)

    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        batch_first = F.embedding(feats, self.weights)
        batch_embedding = F.embedding(feats, self.embedding)
        return self.forward_(batch_size, index, values, self.bias, batch_first,
                             batch_embedding, self.mats)

    @torch.jit.script_method
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.script_method
    def get_type(self):
        return "BIAS_WEIGHT_EMBEDDING_MATS"

    @torch.jit.script_method
    def get_name(self):
        return "AttentionNetMultiHead"


FLAGS = None


def main():
    attention = AttentionNetMultiHead(
        FLAGS.input_dim,
        FLAGS.n_fields,
        FLAGS.embedding_dim,
        FLAGS.num_multi_head,
        FLAGS.top_k,
        FLAGS.num_attention_layers,
        FLAGS.fc_dims)
    attention.save('attention_net_multi_head.pt')


if __name__ == '__main__':
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
        "--num_multi_head",
        type=int,
        default=-1,
        help="num multi head."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="top k."
    )
    parser.add_argument(
        "--num_attention_layers",
        type=int,
        default=-1,
        help="num attention layers."
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
