#!/usr/bin/env python

from __future__ import print_function

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from nn.conv import EdgeSAGEConv


class EdgePropSAGETwoSoftmax(torch.jit.ScriptModule):

    def __init__(self, in_dim, hidden, out_dim, edge_dim, task_type,
                 embedding_dim, field_num, class_weights=""):
        super(EdgePropSAGETwoSoftmax, self).__init__()
        # loss func for multi label classification
        self.loss_fn = torch.nn.BCELoss()
        self.task_type = task_type

        self.input_dim = in_dim
        self.embedding_dim = embedding_dim
        self.n_fields = field_num

        # the transformed dense feature dimension for node
        in_dim = embedding_dim * field_num if embedding_dim > 0 else in_dim

        if len(class_weights) > 2:
            self.class_weights = \
                torch.tensor(list(map(float, class_weights.split(",")))).to(torch.float)
        else:
            self.class_weights = None

        self.gcn1 = EdgeSAGEConv(in_dim, edge_dim, hidden, act=True)
        self.gcn2 = EdgeSAGEConv(hidden, edge_dim, hidden, act=True)

        self.weight = Parameter(torch.zeros(hidden, out_dim))
        self.bias = Parameter(torch.zeros(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias.dim() > 1:
            torch.nn.init.xavier_uniform_(self.bias)

    @torch.jit.script_method
    def forward_(self, x, first_edge_index, second_edge_index,
        first_edge_feature, second_edge_feature):
        embedding = self.embedding_(x, first_edge_index, second_edge_index,
                                    first_edge_feature, second_edge_feature)
        out = torch.matmul(embedding, self.weight)
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
    def predict_(self, x, first_edge_index, second_edge_index,
        first_edge_feature, second_edge_feature):
        output = self.forward_(x, first_edge_index, second_edge_index,
                               first_edge_feature, second_edge_feature)
        if self.task_type == "classification":
            return output.max(1)[1]
        else:
            return output

    @torch.jit.script_method
    def embedding_predict_(self, embedding):
        out = torch.matmul(embedding, self.weight3)
        out = out + self.bias3
        if self.task_type == "classification":
            return F.log_softmax(out, dim=1)
        else:
            return F.sigmoid(out)

    @torch.jit.script_method
    def embedding_(self, x, first_edge_index, second_edge_index,
        first_edge_feature, second_edge_feature):
        # first layer
        out = self.gcn1(x, second_edge_feature, second_edge_index)

        # second layer
        out = self.gcn2(out, first_edge_feature, first_edge_index)

        return out


FLAGS = None


def main():
    sage = EdgePropSAGETwoSoftmax(FLAGS.input_dim, FLAGS.hidden_dim,
                                  FLAGS.output_dim, FLAGS.edge_input_dim,
                                  FLAGS.task_type, FLAGS.input_embedding_dim,
                                  FLAGS.input_field_num, FLAGS.class_weights)
    sage.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="input dimention of node features")
    parser.add_argument(
        "--edge_input_dim",
        type=int,
        default=-1,
        help="input dimention of edge features")
    parser.add_argument(
        "--input_embedding_dim",
        type=int,
        default=-1,
        help="embedding dim of node features")
    parser.add_argument(
        "--input_field_num",
        type=int,
        default=-1,
        help="field num of node features")
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
        default="graphsage.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
