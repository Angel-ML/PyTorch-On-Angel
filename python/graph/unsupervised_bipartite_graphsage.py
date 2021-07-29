#!/usr/bin/env python

from __future__ import print_function

import argparse
import torch
from nn.conv import BiSAGEConv


class UnsupervisedBiSAGE(torch.jit.ScriptModule):

    def __init__(self, in_u_dim, in_i_dim, hidden, out_dim):
        super(UnsupervisedBiSAGE, self).__init__()
        # bipartite graph, two types of nodes, namely u,i. Edges: (u, i)

        self.u_layer1 = BiSAGEConv(in_u_dim, in_i_dim, hidden)
        self.i_layer1 = BiSAGEConv(in_i_dim, in_u_dim, hidden)
        self.ui_layer1 = BiSAGEConv(in_i_dim, hidden, hidden)
        self.iu_layer1 = BiSAGEConv(in_u_dim, hidden, hidden)
        self.u_layer2 = BiSAGEConv(hidden, hidden, out_dim)
        self.i_layer2 = BiSAGEConv(hidden, hidden, out_dim)

    @torch.jit.script_method
    def forward_(self, pos_u, neg_u, pos_i, neg_i, first_u_edge_index,
        second_u_edge_index, first_i_edge_index, second_i_edge_index):
        pos_u_embedding, pos_i_embedding = self.embedding__(pos_u, pos_i,
                                                            first_u_edge_index,
                                                            second_u_edge_index,
                                                            first_i_edge_index,
                                                            second_i_edge_index)
        neg_u_embedding, neg_i_embedding = self.embedding__(neg_u, neg_i,
                                                            first_u_edge_index,
                                                            second_u_edge_index,
                                                            first_i_edge_index,
                                                            second_i_edge_index)

        u_index, i_index = first_u_edge_index[0], first_u_edge_index[1]

        if second_i_edge_index[0][0] != -1:  # case for i-u-i-u
            i_index, u_index = first_i_edge_index[0], first_i_edge_index[1]

        return pos_u_embedding[u_index], pos_i_embedding[i_index], \
               neg_u_embedding[u_index], neg_i_embedding[i_index]

    @torch.jit.script_method
    def loss(self, pos_u_out, pos_i_out, neg_u_out, neg_i_out):
        pos_loss = -torch.log(
            self.discriminate(pos_u_out, pos_i_out, sigmoid=True) + 1e-15).mean()
        neg_loss_u2i = -torch.log(
            1 - self.discriminate(pos_u_out, neg_i_out, sigmoid=True) + 1e-15).mean()
        neg_loss_i2u = -torch.log(
            1 - self.discriminate(pos_i_out, neg_u_out, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss_u2i + neg_loss_i2u

    @torch.jit.script_method
    def discriminate(self, z, neighbor, sigmoid=True):
        # type: (Tensor, Tensor, bool) -> Tensor
        value = torch.sum(torch.mul(z, neighbor), 1)
        return torch.sigmoid(value) if sigmoid else value

    @torch.jit.script_method
    def embedding__(self, u, i, first_u_edge_index, second_u_edge_index,
        first_i_edge_index, second_i_edge_index):
        # sample as u-i-u-i for user embedding
        # sample as i-u-i-u for item embedding
        neigh_u = u
        neigh_i = i

        # first layer
        if second_u_edge_index[0][0] != -1:  # case for u-i-u-i
            neigh_u = self.u_layer1(u, i, second_u_edge_index)
            neigh_i = self.ui_layer1(i, neigh_u, first_i_edge_index)
            neigh_u = self.u_layer2(neigh_u, neigh_i, first_u_edge_index)

        elif second_i_edge_index[0][0] != -1:  # case for i-u-i-u
            neigh_i = self.i_layer1(i, u, second_i_edge_index)
            neigh_u = self.iu_layer1(u, neigh_i, first_u_edge_index)
            neigh_i = self.i_layer2(neigh_i, neigh_u, first_i_edge_index)
        else:  # case for one order
            neigh_u = self.u_layer1(u, i, first_u_edge_index)
            neigh_i = self.i_layer1(i, u, first_i_edge_index)
            return neigh_u, neigh_i

        # second layer
        return neigh_u, neigh_i

    @torch.jit.script_method
    def user_embedding_(self, u, i, first_u_edge_index, second_u_edge_index,
        first_i_edge_index, second_i_edge_index):
        # sample as u-i-u-i
        return self.embedding__(u, i, first_u_edge_index, second_u_edge_index,
                                first_i_edge_index, second_i_edge_index)[0]

    @torch.jit.script_method
    def item_embedding_(self, u, i, first_u_edge_index, second_u_edge_index,
        first_i_edge_index, second_i_edge_index):
        # sample as i-u-i-u
        return self.embedding__(u, i, first_u_edge_index, second_u_edge_index,
                                first_i_edge_index, second_i_edge_index)[1]


FLAGS = None


def main():
    bisage = UnsupervisedBiSAGE(FLAGS.input_user_dim, FLAGS.input_item_dim,
                                FLAGS.hidden_dim, FLAGS.output_dim)
    bisage.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_user_dim",
        type=int,
        default=-1,
        help="input dimention of user node features")
    parser.add_argument(
        "--input_item_dim",
        type=int,
        default=-1,
        help="input dimention of item node features")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=-1,
        help="hidden dimension of bipartite graphsage convolution layer")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=-1,
        help="output embedding dimension")
    parser.add_argument(
        "--output_file",
        type=str,
        default="unsupervised_bipartite_graphsage.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
