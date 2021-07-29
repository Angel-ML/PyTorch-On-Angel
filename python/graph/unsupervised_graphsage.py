#!/usr/bin/env python

from __future__ import print_function

import argparse
import torch
import torch.nn.functional as F
from utils import scatter_mean
from nn.conv import SAGEConv3


class UnsupervisedSAGE(torch.jit.ScriptModule):

    def __init__(self, in_dim, out_dim):
        super(UnsupervisedSAGE, self).__init__()

        self.gcn1 = SAGEConv3(in_dim, out_dim, act=True)
        self.gcn2 = SAGEConv3(out_dim, out_dim)

    @torch.jit.script_method
    def forward_(self, pos_x, neg_x, first_edge_index, second_edge_index):
        pos_embedding, pos_neighbor = self.embedding__(pos_x, first_edge_index,
                                        second_edge_index)
        neg_embedding, neg_neighbor = self.embedding__(neg_x, first_edge_index,
                                        second_edge_index)
        return pos_embedding, pos_neighbor, neg_embedding

    @torch.jit.script_method
    def loss(self, pos_out, pos_neigh_out, neg_out):
        pos_loss = -torch.log(
            self.discriminate(pos_out, pos_neigh_out, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(pos_out, neg_out, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

    @torch.jit.script_method
    def discriminate(self, z, neighbor, sigmoid=True):
        # type: (Tensor, Tensor, bool) -> Tensor
        value = torch.sum(torch.mul(z, neighbor), 1)
        return torch.sigmoid(value) if sigmoid else value

    @torch.jit.script_method
    def embedding__(self, x, first_edge_index, second_edge_index):
        # first layer
        out = self.gcn1(x, second_edge_index)

        # caluate neighbors
        row, col = first_edge_index[0], first_edge_index[1]
        neighbors = scatter_mean(out[col], row, dim=0)  # do not set dim_size

        # second layer
        out = self.gcn2(out, first_edge_index)

        return out, F.normalize(neighbors, p=2.0, dim=-1)

    @torch.jit.script_method
    def embedding_(self, x, first_edge_index, second_edge_index):
        return self.embedding__(x, first_edge_index, second_edge_index)[0]


FLAGS = None


def main():
    sage = UnsupervisedSAGE(FLAGS.input_dim, FLAGS.output_dim)
    sage.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="input dimention of node features")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=-1,
        help="output embedding dimension")
    parser.add_argument(
        "--output_file",
        type=str,
        default="unsupervised_graphsage.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
