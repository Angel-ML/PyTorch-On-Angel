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


class PNN(torch.jit.ScriptModule):

    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, fc_dims=[]):
        super(PNN, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.mats = []
        if input_dim > 0 and embedding_dim > 0 and n_fields > 0 and fc_dims:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1))
            self.weights = torch.nn.Parameter(torch.zeros(input_dim, 1))
            self.embedding = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(input_dim, embedding_dim)))
            product_linear = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.zeros(fc_dims[0], n_fields * embedding_dim)))
            product_bias = torch.nn.Parameter(torch.zeros(1, 1))
            self.mats.append(product_linear)
            self.mats.append(product_bias)
            num_pairs = (int)(n_fields * (n_fields - 1) / 2)
            product_quadratic_inner = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.zeros(fc_dims[0], num_pairs)))
            product_quadratic_inner = torch.jit.annotate(Tensor, product_quadratic_inner)
            self.mats.append(product_quadratic_inner)
            dim = fc_dims[0]
            for fc_dim in fc_dims[1:]:
                w = torch.nn.init.xavier_uniform_(torch.nn.Parameter(torch.zeros(dim, fc_dim)))
                self.mats.append(w)
                b = torch.nn.Parameter(torch.zeros(1, 1))
                self.mats.append(b)
                dim = fc_dim

            self.input_dim = torch.jit.Attribute(self.input_dim, int)
            self.n_fields = torch.jit.Attribute(self.n_fields, int)
            self.embedding_dim = torch.jit.Attribute(self.embedding_dim, int)
            self.mats = torch.jit.Attribute(self.mats, List[Tensor])

    @torch.jit.script_method
    def first_order(self, batch_size, index, values, bias, weights):
        # type: (int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        size = batch_size
        srcs = weights.view(1, -1).mul(values.view(1, -1)).view(-1)
        output = torch.zeros(size, dtype=torch.float32)
        output.scatter_add_(0, index, srcs)
        first = output + bias
        return first

    @torch.jit.script_method
    def product(self, batch_size, embeddings, mats):
        # type: (int, Tensor, List[Tensor]) -> Tensor
        b = batch_size
        bn = embeddings.size(0)
        embedding_dim = embeddings.size(1)
        n_fileds = (int)(bn / b)
        x = embeddings.view(b, -1)
        product_linear_out = x.matmul(mats[0].t())
        x_1 = embeddings.view(b, n_fileds, embedding_dim)
        indices = torch.triu_indices(n_fileds, n_fileds, 1)
        p = torch.index_select(x_1, 1, indices[0])
        q = torch.index_select(x_1, 1, indices[1])
        product_inner_out = torch.sum(p * q, 2) # b * num_pairs
        product_inner_out = product_inner_out.matmul(mats[2].t())
        output = torch.relu(product_linear_out + product_inner_out + mats[1])
        return output

    @torch.jit.script_method
    def deep(self, batch_size, product_input, mats):
        # type: (int, Tensor, List[Tensor]) -> Tensor
        b = batch_size
        output = product_input.view(b, -1)

        for i in range(int(len(mats) / 2)):
            output = torch.relu(output.matmul(mats[i * 2]) + mats[i * 2 + 1])

        return output.view(-1)  # [b * 1]

    @torch.jit.script_method
    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings, mats):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor]) -> Tensor
        first = self.first_order(batch_size, index, values, bias, weights)
        product = self.product(batch_size, embeddings, mats[0:3])
        output = self.deep(batch_size, product, mats[3:])
        return torch.sigmoid(output + first)

    @torch.jit.script_method
    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        batch_first = F.embedding(feats, self.weights)
        emb = F.embedding(feats, self.embedding)
        return self.forward_(batch_size, index, feats, values,
                             self.bias, batch_first, emb, self.mats)

    @torch.jit.script_method
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.script_method
    def get_type(self):
        return "BIAS_WEIGHT_EMBEDDING_MATS"

    @torch.jit.script_method
    def get_name(self):
        return "PNN"


FLAGS = None


def main():
    pnn = PNN(FLAGS.input_dim, FLAGS.n_fields, FLAGS.embedding_dim, FLAGS.fc_dims)
    pnn.save("pnn.pt")


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