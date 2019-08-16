#!/usr/bin/env python

from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from utils import scatter_mean

class SAGETwoSoftmax(torch.jit.ScriptModule):

    def __init__(self, in_dim, hidden, out_dim):
        super(SAGETwoSoftmax, self).__init__()
        self.weight1 = Parameter(torch.zeros(in_dim*2, hidden))
        self.bias1 = Parameter(torch.zeros(hidden))

        self.weight2 = Parameter(torch.zeros(hidden*2, hidden))
        self.bias2 = Parameter(torch.zeros(hidden))

        self.weight3 = Parameter(torch.zeros(hidden, out_dim))
        self.bias3 = Parameter(torch.zeros(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight3)
        if self.bias1.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias1)
        if self.bias2.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias2)
        if self.bias3.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias3)

    @torch.jit.script_method
    def forward_(self, x, first_edge_index, second_edge_index):
        embedding = self.embedding_(x, first_edge_index, second_edge_index)
        out = torch.matmul(embedding, self.weight3)
        out = out + self.bias3
        return F.log_softmax(out, dim=1)

    @torch.jit.script_method
    def loss(self, y_pred, y_true):
        y_true = y_true.view(-1).to(torch.long)
        return F.nll_loss(y_pred, y_true)

    @torch.jit.script_method
    def predict_(self, x, first_edge_index, second_edge_index):
        output = self.forward_(x, first_edge_index, second_edge_index)
        return output.max(1)[1]

    @torch.jit.script_method
    def embedding_(self, x, first_edge_index, second_edge_index):
        # first layer
        row, col = second_edge_index[0], second_edge_index[1]
        out = scatter_mean(x[col], row, dim=0) # do not set dim_size
        out = torch.cat([x[0:out.size(0)], out], dim=1)
        out = torch.matmul(out, self.weight1)
        out = out + self.bias1
        out = torch.relu(out)
        out = F.normalize(out, p=2.0, dim=-1)

        # second layer
        row, col = first_edge_index[0], first_edge_index[1]
        neighbors = scatter_mean(out[col], row, dim=0) # do not set dim_size
        out = torch.cat([out[0:neighbors.size(0)], neighbors], dim=1)
        out = torch.matmul(out, self.weight2)
        out = out + self.bias2
        out = F.normalize(out, p=2.0, dim=-1)
        return out

    def acc(self, y_pred, y_true):
        y_true = y_true.view(-1).to(torch.long)
        return y_pred.max(1)[1].eq(y_true).sum().item() / y_pred.size(0)

FLAGS = None

def main():
    sage = SAGETwoSoftmax(FLAGS.input_dim, FLAGS.hidden_dim, FLAGS.output_dim)
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
        help="hidden dimension of graphsage convolution layer")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=-1,
        help="output dimension, the number of labels")
    parser.add_argument(
        "--output_file",
        type=str,
        default="graphsage.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()