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
from typing import List

import torch
from torch import Tensor


class MMoE(torch.nn.Module):
    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, experts_hidden=-1, experts_out=-1, towers_hidden=-1, towers_out=1, num_experts=6, tasks=1):
        super(MMoE, self).__init__()
        # loss func
        self.loss_fn = torch.nn.BCELoss()
        # input params
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim

        self.experts_out = experts_out
        self.num_experts = num_experts
        self.tasks = tasks

        """Angel Params"""
        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))
        # weights
        self.weights = torch.nn.Parameter(torch.zeros(1, 1))
        # embeddings
        self.embedding = torch.nn.Parameter(torch.zeros(embedding_dim))

        # mats
        self.mats = []
        # experts
        for i in range(num_experts):
            self.mats.append(torch.nn.Parameter(torch.randn(input_dim, experts_hidden)))
            self.mats.append(torch.nn.Parameter(torch.randn(1, experts_hidden)))
            self.mats.append(torch.nn.Parameter(torch.randn(experts_hidden, experts_out)))
            self.mats.append(torch.nn.Parameter(torch.randn(1, experts_out)))
        # gates
        for i in range(tasks):
            self.mats.append(torch.nn.Parameter(torch.randn(input_dim, num_experts)))
        # towers
        for i in range(tasks):
            self.mats.append(torch.nn.Parameter(torch.randn(experts_out, towers_hidden)))
            self.mats.append(torch.nn.Parameter(torch.randn(1, towers_hidden)))
            self.mats.append(torch.nn.Parameter(torch.randn(towers_hidden, towers_out)))
            self.mats.append(torch.nn.Parameter(torch.randn(1, towers_out)))

        # init params
        for i in self.mats:
            torch.nn.init.xavier_uniform_(i)

    def parse_mats(self, mats: List[Tensor]):
        experts_mats = [torch.stack([mats[4 * i + j] for i in range(self.num_experts)]) for j in range(4)]
        gates_mats = [torch.stack([mats[4 * self.num_experts + i] for i in range(self.tasks)])]
        towers_mats = [torch.stack([mats[(4 * self.num_experts + self.tasks) + 4 * i + j] for i in range(self.tasks)]) for j in range(4)]
        return experts_mats, gates_mats, towers_mats

    def experts_module(self, x, experts_mats: List[Tensor]):
        x = torch.relu(torch.baddbmm(experts_mats[1], x.expand(self.num_experts, -1, -1), experts_mats[0]))
        # 6,30,32
        x = torch.nn.functional.dropout(x, p=0.3)
        x = torch.baddbmm(experts_mats[3], x, experts_mats[2])
        # 6,30,16
        return x

    def gates_module(self, x, gates_mats: List[Tensor]):
        x = torch.nn.functional.softmax(torch.bmm(x.expand(self.tasks, -1, -1), gates_mats[0]), dim=2)
        # 2,30,6
        return x

    def towers_input(self, experts_out, gates_out):
        e_o = experts_out.expand(self.tasks, -1, -1, -1)
        # 2,6,30,16
        g_o = gates_out.unsqueeze(3).expand(-1, -1, -1, self.experts_out)
        # 2,30,6,16
        g_o = g_o.permute(0, 2, 1, 3)
        # 2,6,30,16
        x = torch.sum(e_o * g_o, dim=1)
        return x

    def towers_module(self, towers_input, towers_mats: List[Tensor]):
        x = torch.relu(torch.baddbmm(towers_mats[1], towers_input, towers_mats[0]))
        x = torch.nn.functional.dropout(x, p=0.4)
        x = torch.baddbmm(towers_mats[3], x, towers_mats[2])
        x = torch.sigmoid(x)
        return x

    def forward_(self, batch_size: int, index, feats, values, bias, weights, embeddings, mats: List[Tensor], fields=Tensor([])):
        # parse_mats
        experts_mats, gates_mats, towers_mats = self.parse_mats(mats)

        # sparse_coo_tensor
        indices = torch.stack((index, feats), dim=0)
        sparse_x = torch.sparse_coo_tensor(indices, values, size=torch.Size((batch_size, self.input_dim)))
        # to_dense
        dense_x = sparse_x.to_dense()

        # experts_module
        experts_out = self.experts_module(dense_x, experts_mats)
        # gates_module
        gates_out = self.gates_module(dense_x, gates_mats)
        # towers_input
        towers_input = self.towers_input(experts_out, gates_out)
        # towers_module
        pred = self.towers_module(towers_input, towers_mats)
        pred = pred.squeeze(2).permute(1, 0).reshape(-1)
        return pred

    def forward(self, batch_size: int, index, feats, values, fields=Tensor([])):
        return self.forward_(batch_size, index, feats, values, self.bias, self.weights, self.embedding, self.mats, fields)

    @torch.jit.export
    def loss(self, pred, gt):
        pred = pred.view(self.tasks, -1)
        gt = gt.view(self.tasks, -1)
        return self.loss_fn(pred, gt)

    @torch.jit.export
    def get_type(self):
        return "BIAS_WEIGHT_EMBEDDING_MATS"

    @torch.jit.export
    def get_name(self):
        return "MMoE"


FLAGS = None


def main():
    mmoe = MMoE(
        input_dim=FLAGS.input_dim,
        n_fields=FLAGS.n_fields,
        embedding_dim=FLAGS.embedding_dim,
        experts_hidden=FLAGS.experts_hidden,
        experts_out=FLAGS.experts_out,
        towers_hidden=FLAGS.towers_hidden,
        towers_out=1,
        num_experts=FLAGS.num_experts,
        tasks=FLAGS.tasks,
    )
    mmoe_script_module = torch.jit.script(mmoe)
    mmoe_script_module.save("MMoE.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=148,
        help="data input dim.",
    )
    parser.add_argument(
        "--n_fields",
        type=int,
        default=-1,
        help="data num fields.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=1,
        help="embedding dim.",
    )
    parser.add_argument(
        "--experts_hidden",
        type=int,
        default=32,
        help="experts hidden.",
    )
    parser.add_argument(
        "--experts_out",
        type=int,
        default=16,
        help="experts out.",
    )
    parser.add_argument(
        "--towers_hidden",
        type=int,
        default=8,
        help="towers hidden.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=6,
        help="num experts.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=2,
        help="num tasks.",
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
