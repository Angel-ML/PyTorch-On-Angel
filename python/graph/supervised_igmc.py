#!/usr/bin/env python

from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Parameter
from nn.conv import IGMCConv


class SupervisedIGMC(torch.jit.ScriptModule):

    def __init__(self, in_u_dim, in_i_dim, hidden, edge_types, out_dim,
        n_latent, dropout, n_bases, method, arr_lambda):
        super(SupervisedIGMC, self).__init__()
        # bipartite graph, two types of nodes, namely u,i. Edges: (u, i)
        self.weightU = Parameter(
            torch.zeros(in_u_dim, hidden))
        self.weightI = Parameter(
            torch.zeros(in_i_dim, hidden))

        self.weight = Parameter(torch.zeros(128, out_dim))
        self.bias = Parameter(torch.zeros(out_dim))
        self.dropout = dropout
        self.method = method
        self.arr_lambda = arr_lambda

        self.convs = torch.nn.ModuleList()
        self.latent_dim = [hidden] * n_latent
        self.convs.append(IGMCConv(self.latent_dim[0], n_bases, edge_types))
        for i in range(0, len(self.latent_dim) - 1):
            self.convs.append(
                IGMCConv(self.latent_dim[i + 1], n_bases, edge_types))
        self.lin1 = Linear(2 * sum(self.latent_dim) + in_u_dim + in_i_dim, 128)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weightU)
        torch.nn.init.xavier_uniform_(self.weightI)
        self.lin1.reset_parameters()

        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

    @torch.jit.script_method
    def forward_(self, labeled_edge_index, edge_index, pos_u, pos_i, edge_type):
        out = self.embedding_(labeled_edge_index, edge_index, pos_u, pos_i, edge_type)

        out = torch.matmul(out, self.weight)
        out = out + self.bias
        if self.method == "classification":
            out = F.log_softmax(out, dim=-1)
        else:
            out = out[:, 0]
        return out

    @torch.jit.script_method
    def loss(self, y_pred, y_true):
        if self.method == "classification":
            y_true = y_true.view(-1).to(torch.long)
            loss = F.nll_loss(y_pred, y_true)
        else:
            y_true = y_true.to(torch.float)
            y_pred = y_pred.to(torch.float)
            loss = F.mse_loss(y_pred, y_true)

            if self.arr_lambda > 0:
                for gconv in self.convs:
                    w = torch.matmul(
                        gconv.att,
                        gconv.basis.view(gconv.n_bases, -1)
                    ).view(gconv.edge_types, gconv.out_h, gconv.out_h)
                    reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
                    loss += self.arr_lambda * reg_loss
        return loss

    @torch.jit.script_method
    def predict_(self, labeled_edge_index, edge_index, pos_u, pos_i, edge_type):
        if self.method == "classification":
            output = self.forward_(labeled_edge_index, edge_index,
                                   pos_u, pos_i, edge_type).max(1)[1]
        else:
            output = self.forward_(labeled_edge_index, edge_index,
                                   pos_u, pos_i, edge_type)
        return output

    @torch.jit.script_method
    def embedding_(self, labeled_edge_index, edge_index, pos_u, pos_i, edge_type):
        row, col = labeled_edge_index[0], labeled_edge_index[1]

        concat_states_u = []
        concat_states_i = []
        u = torch.matmul(pos_u, self.weightU)
        i = torch.matmul(pos_i, self.weightI)
        for conv in self.convs:
            u_embedding, i_embedding = conv(u, i, edge_index, edge_type)
            concat_states_u.append(u_embedding)
            concat_states_i.append(i_embedding)
        concat_states_u = torch.cat(concat_states_u, 1)
        concat_states_i = torch.cat(concat_states_i, 1)
        out = torch.cat([concat_states_u[row], concat_states_i[col]], dim=1)

        out = torch.cat([out, pos_u[row], pos_i[col]], 1)
        out = torch.relu(self.lin1(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    @torch.jit.script_method
    def embedding_predict_(self, embedding):
        out = torch.matmul(embedding, self.weight)
        out = out + self.bias
        if self.method == "classification":
            out = F.log_softmax(out, dim=-1)
        else:
            out = out[:, 0]
        return out


FLAGS = None


def main():
    igmc = SupervisedIGMC(FLAGS.input_user_dim, FLAGS.input_item_dim,
                          FLAGS.hidden_dim, FLAGS.edge_types, FLAGS.output_dim,
                          FLAGS.n_latent, FLAGS.dropout, FLAGS.n_bases,
                          FLAGS.method, FLAGS.arr_lambda)
    igmc.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_user_dim",
        type=int,
        default=-1,
        help="input dimension of user node features")
    parser.add_argument(
        "--input_item_dim",
        type=int,
        default=-1,
        help="input dimension of item node features")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=-1,
        help="hidden dimension of convolution layer")
    parser.add_argument(
        "--edge_types",
        type=int,
        default=-1,
        help="the num of edge_types")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=-1,
        help="output embedding dimension")
    parser.add_argument(
        "--n_latent",
        type=int,
        default=1,
        help="num of latent layer")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="num of latent layer")
    parser.add_argument(
        "--n_bases",
        type=int,
        default=30,
        help="hidden dimension of weight for edge_types convolution layer")
    parser.add_argument(
        "--method",
        type=str,
        default="classification",
        help="classification or regression")
    parser.add_argument(
        "--arr_lambda",
        type=float,
        default=0,
        help="trades-off the importance of MSE loss and ARR regularizer")
    parser.add_argument(
        "--output_file",
        type=str,
        default="supervised_igmc_classification.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
