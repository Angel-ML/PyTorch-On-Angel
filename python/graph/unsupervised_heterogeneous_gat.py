#!/usr/bin/env python

from __future__ import print_function

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from nn.conv import HGATConv
from utils import parse_feat


class UnsupervisedHGAT(torch.jit.ScriptModule):

    def __init__(self, in_u_dim, in_i_dim, u_embedding_dim, i_embedding_dim,
                 u_field_num, i_field_num, hidden, out_dim, heads, dropout,
                 negative_size, encode):
        super(UnsupervisedHGAT, self).__init__()
        # bipartite graph, two types of nodes, namely u,i. Edges: (u, i)

        self.user_input_dim = in_u_dim
        self.item_input_dim = in_i_dim
        self.user_embedding_dim = u_embedding_dim
        self.item_embedding_dim = i_embedding_dim
        self.user_field_num = u_field_num
        self.item_field_num = i_field_num
        self.heads = heads
        self.dropout = dropout
        self.negative_size = negative_size
        self.encode = encode

        # the transformed dense feature dimension for user and item
        in_u_dim = u_embedding_dim * u_field_num if u_embedding_dim > 0 else in_u_dim
        in_i_dim = i_embedding_dim * i_field_num if i_embedding_dim > 0 else in_i_dim

        self.convs_u = torch.nn.ModuleList()
        self.convs_i = torch.nn.ModuleList()

        # u-i-u
        self.convs_u.append(HGATConv(hidden, hidden, hidden, heads=heads, dropout=dropout))
        self.convs_u.append(HGATConv(hidden * self.heads, hidden * self.heads,
                                     hidden, heads=heads, dropout=dropout, act=True))

        # i-u-i
        self.convs_i.append(HGATConv(hidden, hidden, hidden, heads=heads, dropout=dropout))
        self.convs_i.append(HGATConv(hidden * self.heads, hidden * self.heads,
                                     hidden, heads=heads, dropout=dropout, act=True))

        # weight
        self.weight_u1 = Parameter(torch.zeros(in_u_dim, hidden))
        self.weight_i1 = Parameter(torch.zeros(in_i_dim, hidden))

        self.weight_u2 = Parameter(torch.zeros(hidden + hidden * self.heads * 2, out_dim))
        self.weight_i2 = Parameter(torch.zeros(hidden + hidden * self.heads * 2, out_dim))

        self.weight = Parameter(torch.zeros(out_dim, 2))
        self.bias = Parameter(torch.zeros(2))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_u1)
        torch.nn.init.xavier_uniform_(self.weight_i1)
        torch.nn.init.xavier_uniform_(self.weight_u2)
        torch.nn.init.xavier_uniform_(self.weight_i2)
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

    @torch.jit.script_method
    def forward_(self, pos_u, pos_i, first_u_edge_index, second_u_edge_index,
                 first_i_edge_index, second_i_edge_index,
                 u_batch_ids=torch.Tensor([]), i_batch_ids=torch.Tensor([]),
                 u_field_ids=torch.Tensor([]), i_field_ids=torch.Tensor([])):

        # encode u, i feat
        pos_u, pos_i = self.encode_feat(pos_u, pos_i, u_batch_ids,
                                        i_batch_ids, u_field_ids, i_field_ids)

        pos_u_embedding = self.embedding__(pos_u, pos_i,
                                           first_u_edge_index,
                                           first_i_edge_index,
                                           "u")
        pos_i_embedding = self.embedding__(pos_i, pos_u,
                                           second_u_edge_index,
                                           second_i_edge_index,
                                           "i")

        u_index, i_index = first_u_edge_index[0], first_u_edge_index[1]

        u_emb, i_emb = pos_u_embedding[u_index], pos_i_embedding[i_index]

        # out size: (edge_num, out_dim)
        return u_emb, i_emb

    @torch.jit.script_method
    def encode_feat(self, u, i,
                    u_batch_ids=torch.Tensor([]), i_batch_ids=torch.Tensor([]),
                    u_field_ids=torch.Tensor([]), i_field_ids=torch.Tensor([])):

        if self.user_field_num > 0:
            u = parse_feat(u, u_batch_ids, u_field_ids, self.user_field_num, self.encode)
            i = parse_feat(i, i_batch_ids, i_field_ids, self.item_field_num, self.encode)

        u = torch.matmul(u, self.weight_u1)
        i = torch.matmul(i, self.weight_i1)

        return u, i

    @torch.jit.script_method
    def loss(self, pos_u, pos_i):
        # (edge_num, out_dim)
        i_emb_origin = pos_i.repeat(1, 1)
        i_emb = pos_i
        # ((edge_num * (negative_size + 1)), out_dim)
        u_emb = pos_u.repeat(self.negative_size + 1, 1)

        for _ in range(self.negative_size):
            neg_index = torch.randperm(i_emb_origin.size(0)).view(-1).to(torch.long)
            neg_sample_emb = torch.index_select(i_emb_origin, 0, neg_index)
            i_emb = torch.cat([i_emb, neg_sample_emb], dim=0)

        # ((edge_num * (negative_size + 1)), 1)
        prod = torch.sum(torch.mul(u_emb, i_emb), 1, True) / 0.05

        # (edge_num, (negative_size + 1))
        logits = torch.transpose(torch.transpose(prod, 0, 1)
                                 .reshape(self.negative_size + 1, -1), 0, 1)

        true_label = torch.zeros(pos_u.size(0)).view(-1).to(torch.long)

        loss = F.cross_entropy(logits, true_label)

        acc_per_example = torch.eq(
            torch.argmax(logits, dim=1),
            torch.zeros([pos_u.size(0)], dtype=torch.int64)).view(-1).to(torch.float)

        loss1 = loss.mean()
        accuracy = acc_per_example.mean()
        print("loss: ", loss1, ", acc: ", accuracy)

        return loss

    @torch.jit.script_method
    def embedding__(self, u, i, first_u_edge_index, first_i_edge_index, conv="u"):
        # type: (Tensor, Tensor, Tensor, Tensor, str) -> Tensor

        # gat conv encode
        graph_user_emb = self.encode_conv(u, i, first_u_edge_index, first_i_edge_index, conv)

        user_emb = torch.cat([u[0: graph_user_emb.size(0)], graph_user_emb], dim=1)

        # conv for u, i
        if conv == "u":
            user_emb = torch.matmul(user_emb, self.weight_u2)
        else:
            user_emb = torch.matmul(user_emb, self.weight_i2)

        user_emb = F.normalize(user_emb, p=2.0)

        # out size: (node_num, out_dim)
        return user_emb

    @torch.jit.script_method
    def encode_conv(self, u, i, first_edge_index, second_edge_index, conv):
        # type: (Tensor, Tensor, Tensor, Tensor, str) -> Tensor

        # convs = self.convs_u
        if conv == "u":
            convs = self.convs_u
            z1 = convs[0](i, u, second_edge_index)
            z2 = convs[0](u, i, first_edge_index)
            z3 = convs[1](z2, z1, first_edge_index)
        else:
            convs = self.convs_i
            z1 = convs[0](i, u, second_edge_index)
            z2 = convs[0](u, i, first_edge_index)
            z3 = convs[1](z2, z1, first_edge_index)

        midh0 = torch.cat([z2, z3], dim=1)

        # out size: (node_num, (hidden * heads * 2))
        return midh0

    @torch.jit.script_method
    def user_embedding_(self, u, i, first_u_edge_index, first_i_edge_index,
                        u_batch_ids=torch.Tensor([]), i_batch_ids=torch.Tensor([]),
                        u_field_ids=torch.Tensor([]), i_field_ids=torch.Tensor([])):

        if self.user_field_num > 0:
            u, i = self.encode_feat(u, i, u_batch_ids, i_batch_ids, u_field_ids, i_field_ids)

        user_emb = self.embedding__(u, i, first_u_edge_index, first_i_edge_index, "u")

        # out size: (node_num, out_dim)
        return user_emb

    @torch.jit.script_method
    def item_embedding_(self, u, i, first_u_edge_index, first_i_edge_index,
                        u_batch_ids=torch.Tensor([]), i_batch_ids=torch.Tensor([]),
                        u_field_ids=torch.Tensor([]), i_field_ids=torch.Tensor([])):

        if self.user_field_num > 0:
            u, i = self.encode_feat(u, i, u_batch_ids, i_batch_ids, u_field_ids, i_field_ids)

        item_emb = self.embedding__(i, u, first_u_edge_index, first_i_edge_index, "i")

        # out size: (node_num, out_dim)
        return item_emb


FLAGS = None


def main():
    bisage = UnsupervisedHGAT(FLAGS.input_user_dim, FLAGS.input_item_dim,
                              FLAGS.input_user_embedding_dim,
                              FLAGS.input_item_embedding_dim,
                              FLAGS.input_user_field_num,
                              FLAGS.input_item_field_num,
                              FLAGS.hidden_dim, FLAGS.output_dim,
                              FLAGS.heads, FLAGS.dropout, FLAGS.negative_size,
                              FLAGS.encode)
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
        "--input_user_embedding_dim",
        type=int,
        default=-1,
        help="embedding dim of user node features")
    parser.add_argument(
        "--input_item_embedding_dim",
        type=int,
        default=-1,
        help="embedding dim of item node features")
    parser.add_argument(
        "--input_user_field_num",
        type=int,
        default=-1,
        help="field num of user node features")
    parser.add_argument(
        "--input_item_field_num",
        type=int,
        default=-1,
        help="field num of item node features")
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
        "--heads",
        type=int,
        default=1,
        help="the number of heads")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="the percentage of dropout")
    parser.add_argument(
        "--negative_size",
        type=int,
        default=1,
        help="the negative sample size")
    parser.add_argument(
        "--encode",
        type=str,
        default="dense",
        help="data encode, could be one-hot, multi-hot or dense")
    parser.add_argument(
        "--output_file",
        type=str,
        default="unsupervised_bipartite_graphsage.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
