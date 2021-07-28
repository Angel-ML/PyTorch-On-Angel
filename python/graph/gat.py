#!/usr/bin/env python

from __future__ import print_function

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from nn.conv import GATConv


class GATTwoOrder(torch.jit.ScriptModule):

    def __init__(self, in_dim, hidden, out_dim, heads, dropout, task_type,
        class_weights=""):
        super(GATTwoOrder, self).__init__()
        # loss func for multi label classification
        self.loss_fn = torch.nn.BCELoss()
        self.task_type = task_type

        if len(class_weights) > 2:
            self.class_weights = \
                torch.tensor(list(map(float, class_weights.split(",")))).to(torch.float)
        else:
            self.class_weights = None

        self.gat1 = GATConv(in_dim, hidden, heads, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, dropout=dropout)
        self.hidden = hidden
        self.heads = heads

        self.weight = Parameter(torch.zeros(hidden, out_dim))
        self.bias = Parameter(torch.zeros(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

    @torch.jit.script_method
    def forward_(self, x, first_edge_index, second_edge_index=torch.Tensor([])):
        x = self.embedding_(x, first_edge_index, second_edge_index)
        out = torch.matmul(x, self.weight)
        out = out + self.bias
        if self.task_type == "classification":
            return F.log_softmax(out, dim=1)
        else:
            return F.sigmoid(out)

    @torch.jit.script_method
    def loss(self, y_pred, y_true):
        if self.task_type == "classification":
            y_true = y_true.view(-1).to(torch.long)
            if self.class_weights is None:
                return F.nll_loss(y_pred, y_true)
            else:
                return F.nll_loss(y_pred, y_true, weight=self.class_weights)
        else:
            u_true = y_true.reshape(y_pred.size())
            return self.loss_fn(y_pred, u_true)

    @torch.jit.script_method
    def predict_(self, x, first_edge_index, second_edge_index=torch.Tensor([])):
        output = self.forward_(x, first_edge_index, second_edge_index)
        if self.task_type == "classification":
            return output.max(1)[1]
        else:
            return output

    @torch.jit.script_method
    def embedding_predict_(self, embedding):
        out = torch.matmul(embedding, self.weight)
        out = out + self.bias
        if self.task_type == "classification":
            return F.log_softmax(out, dim=1)
        else:
            return F.sigmoid(out)

    @torch.jit.script_method
    def embedding_(self, x, first_edge_index, second_edge_index=torch.Tensor([])):
        # first layer
        if second_edge_index.size(0) == 0:
            x = self.gat1(x, first_edge_index)
        else:
            x = self.gat1(x, second_edge_index)
            x = self.gat2(x, first_edge_index)
        return x


FLAGS = None


def main():
    sage = GATTwoOrder(FLAGS.input_dim, FLAGS.hidden_dim, FLAGS.output_dim,
                       FLAGS.heads, FLAGS.dropout, FLAGS.task_type,
                       FLAGS.class_weights)
    sage.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="input dimention of node features")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=-1,
        help="hidden dimension of gat convolution layer")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=-1,
        help="output dimension, the number of labels")
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
        "--task_type",
        type=str,
        default="classification",
        help="classification or multi-label-classification")
    parser.add_argument(
        "--class_weights",
        type=str,
        default="",
        help="class weights, in order to balance class, such as: 0.1,0.9")
    parser.add_argument(
        "--output_file",
        type=str,
        default="gat.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
