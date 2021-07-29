#!/usr/bin/env python

from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from nn.conv import BiSAGEConv
from utils import parse_feat


class SupervisedBiSAGE(torch.jit.ScriptModule):

    def __init__(self, in_u_dim, user_embedding_dim, user_field_num,
        in_i_dim, item_embedding_dim, item_field_num, hidden,
        out_dim, task_type, encode):
        super(SupervisedBiSAGE, self).__init__()
        # bipartite graph, two types of nodes, namely u,i. Edges: (u, i)
        self.loss_fn = torch.nn.BCELoss()
        self.task_type = task_type
        self.encode = encode

        self.user_input_dim = in_u_dim
        self.item_input_dim = in_i_dim
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.user_field_num = user_field_num
        self.item_field_num = item_field_num

        # the transformed dense feature dimension for user and item
        in_u_dim = user_embedding_dim * user_field_num if user_embedding_dim > 0 else in_u_dim
        in_i_dim = item_embedding_dim * item_field_num if item_embedding_dim > 0 else in_i_dim

        self.u_layer2 = BiSAGEConv(in_u_dim, in_i_dim, hidden)
        self.i_layer1 = BiSAGEConv(in_i_dim, in_u_dim, hidden)
        self.u_layer1 = BiSAGEConv(hidden, hidden, hidden)

        self.weight = Parameter(torch.zeros(hidden, out_dim))
        self.bias = Parameter(torch.zeros(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

    @torch.jit.script_method
    def forward_(self, pos_u, pos_i, first_u_edge_index,
        second_u_edge_index=torch.Tensor([]), first_i_edge_index=torch.Tensor([]),
        u_batch_ids=torch.Tensor([]), i_batch_ids=torch.Tensor([]),
        u_field_ids=torch.Tensor([]), i_field_ids=torch.Tensor([])):
        pos_u_embedding = self.embedding_(pos_u, pos_i, first_u_edge_index,
                                          second_u_edge_index, first_i_edge_index,
                                          u_batch_ids, i_batch_ids,
                                          u_field_ids, i_field_ids)
        out = torch.matmul(pos_u_embedding, self.weight)
        out = out + self.bias
        if self.task_type == "classification":
            return F.log_softmax(out, dim=1)
        else:
            return F.sigmoid(out)

    @torch.jit.script_method
    def loss(self, u_pred, u_true):
        if self.task_type == "classification":
            u_true = u_true.view(-1).to(torch.long)
            return F.nll_loss(u_pred, u_true)
        else:
            u_true = u_true.reshape(u_pred.size())
            return self.loss_fn(u_pred, u_true)

    @torch.jit.script_method
    def predict_(self, pos_u, pos_i, first_u_edge_index,
        second_u_edge_index=torch.Tensor([]), first_i_edge_index=torch.Tensor([]),
        u_batch_ids=torch.Tensor([]), i_batch_ids=torch.Tensor([]),
        u_field_ids=torch.Tensor([]), i_field_ids=torch.Tensor([])):
        output = self.forward_(pos_u, pos_i, first_u_edge_index,
                               second_u_edge_index, first_i_edge_index,
                               u_batch_ids, i_batch_ids,
                               u_field_ids, i_field_ids)
        if self.task_type == "classification":
            return output.max(1)[1]
        else:
            return output

    @torch.jit.script_method
    def embedding_(self, u, i, first_u_edge_index,
        second_u_edge_index=torch.Tensor([]), first_i_edge_index=torch.Tensor([]),
        u_batch_ids=torch.Tensor([]), i_batch_ids=torch.Tensor([]),
        u_field_ids=torch.Tensor([]), i_field_ids=torch.Tensor([])):

        if self.user_field_num > 0:
            u = parse_feat(u, u_batch_ids, u_field_ids, self.user_field_num, self.encode)
            i = parse_feat(i, i_batch_ids, i_field_ids, self.item_field_num, self.encode)

        if second_u_edge_index.size(0) == 0:
            neigh_u_second = self.u_layer2(u, i, second_u_edge_index)
            neigh_i = self.i_layer1(i, neigh_u_second, first_i_edge_index)
            neigh_u = self.u_layer1(neigh_u_second, neigh_i, first_u_edge_index)
        else:
            neigh_u = self.u_layer2(u, i, first_u_edge_index)
        return neigh_u

    @torch.jit.script_method
    def embedding_predict_(self, embedding):
        out = torch.matmul(embedding, self.weight)
        out = out + self.bias
        if self.task_type == "classification":
            return F.log_softmax(out, dim=1)
        else:
            return F.sigmoid(out)


FLAGS = None


def main():
    bisage = SupervisedBiSAGE(FLAGS.input_user_dim, FLAGS.input_user_embedding_dim,
                              FLAGS.input_user_field_num, FLAGS.input_item_dim,
                              FLAGS.input_item_embedding_dim, FLAGS.input_item_field_num,
                              FLAGS.hidden_dim, FLAGS.output_dim, FLAGS.task_type, FLAGS.encode)
    bisage.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="classification",
                        help="classification or multi-label-classification.")
    parser.add_argument(
        "--input_user_dim",
        type=int,
        default=-1,
        help="input dimention of user node features")
    parser.add_argument(
        "--input_user_embedding_dim",
        type=int,
        default=-1,
        help="embedding dim of user node features")
    parser.add_argument(
        "--input_user_field_num",
        type=int,
        default=-1,
        help="field num of user node features")
    parser.add_argument(
        "--input_item_dim",
        type=int,
        default=-1,
        help="input dimention of item node features")
    parser.add_argument(
        "--input_item_embedding_dim",
        type=int,
        default=-1,
        help="embedding dim of item node features")
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
        "--encode",
        type=str,
        default="dense",
        help="data encode, could be one-hot, multi-hot or dense")
    parser.add_argument(
        "--output_file",
        type=str,
        default="semi_bipartite_graphsage.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
