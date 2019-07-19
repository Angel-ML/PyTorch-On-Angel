#!/usr/bin/env python

from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F

from torch import Tensor
from typing import List


class xDeepFM(torch.jit.ScriptModule):

    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, fc_dims=[], cin_dims=[]):
        super(xDeepFM, self).__init__()

        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.fc_dims = fc_dims
        self.cin_dims = cin_dims
        self.mats = []

        if input_dim > 0 and n_fields > 0 and embedding_dim > 0 and fc_dims and cin_dims:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1))
            self.weights = torch.nn.Parameter(torch.zeros(input_dim, 1))
            self.embedding = torch.nn.Parameter(torch.zeros(input_dim, embedding_dim))
            torch.nn.init.xavier_uniform_(self.weights)
            torch.nn.init.xavier_uniform_(self.embedding)

            # cin
            n_fields_list = [n_fields]
            for i, size in enumerate(self.cin_dims):
                w = torch.nn.init.xavier_uniform_(torch.nn.Parameter(
                    torch.zeros(size, n_fields_list[-1] * n_fields_list[0], 1)))
                b = torch.nn.Parameter(torch.zeros(size, 1))
                self.mats.append(w.view(size, n_fields_list[-1] * n_fields_list[0]))
                self.mats.append(b)
                n_fields_list.append(size)

            # mlps
            dim = n_fields * embedding_dim
            for (index, fc_dim) in enumerate(fc_dims):
                self.mats.append(torch.nn.Parameter(torch.randn(dim, fc_dim)))
                self.mats.append(torch.nn.Parameter(torch.randn(1, 1)))
                torch.nn.init.xavier_uniform_(self.mats[index * 2])
                dim = fc_dim

            w = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(fc_dims[-1] + sum(cin_dims), 1)))
            self.mats.append(w.view(-1, 1))
            self.mats.append(torch.nn.Parameter(torch.zeros(1, 1)))

            self.input_dim = torch.jit.Attribute(self.input_dim, int)
            self.n_fields = torch.jit.Attribute(self.n_fields, int)
            self.embedding_dim = torch.jit.Attribute(self.embedding_dim, int)
            self.fc_dims = torch.jit.Attribute(self.fc_dims, List[int])
            self.cin_dims = torch.jit.Attribute(self.cin_dims, List[int])
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
    def cin(self, batch_size, embeddings, mats):
        # type: (int, Tensor, List[Tensor]) -> Tensor
        b = batch_size
        n_fields = (int)(embeddings.size(0) / b)
        embedding_dim = embeddings.size(1)
        x0 = embeddings.view(b, n_fields, embedding_dim)
        results = []
        xk = x0
        for i in range(len(self.cin_dims)):
            z = torch.einsum('bhd,bmd->bhmd', xk, x0)
            z = z.view(b, xk.shape[1] * n_fields, embedding_dim) # b * hk * d
            filter_w = mats[i * 2].view(self.cin_dims[i], xk.shape[1] * n_fields, 1)
            filter_b = mats[i * 2 + 1].view(self.cin_dims[i])
            x_out = F.conv1d(z, filter_w, filter_b)
            x_out = torch.relu(x_out)
            next_hidden, res = x_out, x_out
            xk = next_hidden
            results.append(res)
        final_result = torch.cat(results, dim=1)
        final_result = torch.sum(final_result, dim=2)
        return final_result.view(b, -1)

    @torch.jit.script_method
    def deep(self, batch_size, embeddings, mats):
        # type: (int, Tensor, List[Tensor]) -> Tensor
        b = batch_size
        output = embeddings.view(b, -1)

        for i in range(int(len(mats) / 2)):
            output = torch.relu(output.matmul(mats[i * 2]) + mats[i * 2 + 1])

        return output.view(b, -1)  # [b * 1]


    @torch.jit.script_method
    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings, mats):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor]) -> Tensor
        first = self.first_order(batch_size, index, values, bias, weights)
        cin_index = len(self.cin_dims) * 2
        cin = self.cin(batch_size, embeddings, mats[0:cin_index])
        deep = self.deep(batch_size, embeddings, mats[cin_index:-2])
        cin_and_deep = torch.cat([cin, deep], dim=1)
        output = torch.matmul(cin_and_deep, mats[-2]).view(-1)
        output = output + first + mats[-1].view(-1)

        return torch.sigmoid(output)

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
        return "xDeepFM"


FLAGS = None


def main():
    xdeepfm = xDeepFM(FLAGS.input_dim, FLAGS.n_fields, FLAGS.embedding_dim, FLAGS.fc_dims, FLAGS.cin_dims)
    xdeepfm.save("xdeepfm.pt")


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
    parser.add_argument(
        "--cin_dims",
        nargs="+",
        type=int,
        default=-1,
        help="cin layers dim list."
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()