# !/usr/bin/env python

from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F

from torch import Tensor
from typing import List

class DCNet(torch.jit.ScriptModule):
    """
    x_0 * x_l^T * w_l + x_l + b_l
    """
    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, cross_depth=-1, fc_dims=[]):
        super(DCNet, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.cross_depth = cross_depth
        self.mats = []

        if input_dim > 0 and n_fields > 0 and embedding_dim > 0 and cross_depth > 0 and fc_dims:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1))
            self.weights = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(input_dim, 1)))
            self.embedding = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(input_dim, embedding_dim)))

            for i in range(cross_depth):
                wx = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_fields, embedding_dim)))
                self.mats.append(wx.view(-1, 1))
                self.mats.append(torch.nn.Parameter(torch.zeros(1, 1)))

            x_dim = n_fields * embedding_dim
            dim = x_dim
            for fc_dim in fc_dims:
                w = torch.nn.init.xavier_uniform_(torch.nn.Parameter(torch.zeros(dim, fc_dim)))
                self.mats.append(w)
                self.mats.append(torch.nn.Parameter(torch.zeros(1, 1)))
                dim = fc_dim

            w = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(dim+x_dim, 1)))
            self.mats.append(w.view(-1, 1))
            self.mats.append(torch.nn.Parameter(torch.zeros(1, 1)))

            self.input_dim = torch.jit.Attribute(self.input_dim, int)
            self.n_fields = torch.jit.Attribute(self.n_fields, int)
            self.embedding_dim = torch.jit.Attribute(self.embedding_dim, int)
            self.cross_depth = torch.jit.Attribute(self.cross_depth, int)
            self.mats = torch.jit.Attribute(self.mats, List[Tensor])

    @torch.jit.script_method
    def first_order(self, batch_size, index, values, bias, weights):
        # type: (int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        size = batch_size
        srcs = weights.view(1, -1).mul(values.view(1, -1)).view(-1)
        output = torch.zeros(size, dtype=torch.float32)
        output.scatter_add_(0, index, srcs)
        first = output + bias
        return first.view(-1)

    @torch.jit.script_method
    def cross(self, batch_size, embeddings, mats):
        # type: (int, Tensor, List[Tensor]) -> Tensor

        # embedings: [n, embedding_dim] = [b, n_field, embedding_dim]
        b = batch_size
        x0 = embeddings.view(b, -1)
        xl = x0  # x0: [b * x_dim], mats: [x_dim]
        for i in range(int(len(mats)/2)):
            t = xl.matmul(mats[i * 2]).view(b, -1)
            t = x0.mul(t)
            t = t + mats[i * 2 + 1] + xl
            xl = t
        return xl.view(b, -1) # [b * x_dim]

    @torch.jit.script_method
    def higher_order(self, batch_size, embeddings, mats):
        # type: (int, Tensor, List[Tensor]) -> Tensor
        b = batch_size
        output = embeddings.view(b, -1)

        for i in range(int(len(mats) / 2)):
            output = torch.relu(output.matmul(mats[i * 2]) + mats[i * 2 + 1])

        return output.view(b, -1) # [b * 1]


    @torch.jit.script_method
    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings, mats):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor]) -> Tensor
        first = self.first_order(batch_size, index, values, bias, weights)
        cross_mats_index = self.cross_depth * 2
        cross = self.cross(batch_size, embeddings, mats[0:cross_mats_index])
        higher = self.higher_order(batch_size, embeddings, mats[cross_mats_index:-2])
        cross_and_higher = torch.cat([cross, higher], dim=1)
        output = torch.matmul(cross_and_higher, mats[-2]).view(-1)
        output = output + first + mats[-1].view(-1)
        return torch.sigmoid(output)

    @torch.jit.script_method
    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        emb = F.embedding(feats, self.embedding)
        first = F.embedding(feats, self.weights)
        return self.forward_(batch_size, index, feats, values,
                             self.bias, first, emb, self.mats)

    @torch.jit.script_method
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.script_method
    def get_type(self):
        return "BIAS_WEIGHT_EMBEDDING_MATS"

    @torch.jit.script_method
    def get_name(self):
        return "DCNet"


FLAGS = None


def main():
    dcn = DCNet(FLAGS.input_dim, FLAGS.n_fields, FLAGS.embedding_dim, FLAGS.cross_depth, FLAGS.fc_dims)
    dcn.save("dcn.pt")


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
        "--cross_depth",
        type=int,
        default=-1,
        help="cross layers depth."
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