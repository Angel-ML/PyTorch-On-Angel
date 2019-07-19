#!/usr/bin/env python

from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F


## model
class LogisticRegression(torch.jit.ScriptModule):

    def __init__(self, input_dim=-1):
        super(LogisticRegression, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim

        if input_dim > 0:
            self.bias = torch.zeros(1, 1)
            self.weights = torch.randn(input_dim, 1)
            self.bias = torch.nn.Parameter(self.bias, requires_grad=True)
            self.weights = torch.nn.Parameter(self.weights, requires_grad=True)
            torch.nn.init.xavier_uniform_(self.weights)

            self.input_dim = torch.jit.Attribute(self.input_dim, int)


    @torch.jit.script_method
    def forward_(self, batch_size, index, feats, values, bias, weight):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
        size = batch_size
        index = index.view(-1)
        values = values.view(1, -1)
        srcs = weight.view(1, -1).mul(values).view(-1)
        output = torch.zeros(size, dtype=torch.float32)
        output.scatter_add_(0, index, srcs)
        output = output + bias
        return torch.sigmoid(output)

    @torch.jit.script_method
    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        weight = F.embedding(feats, self.weights)
        bias = self.bias
        return self.forward_(batch_size, index, feats, values, bias, weight)

    @torch.jit.script_method
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.script_method
    def get_type(self):
        return "BIAS_WEIGHT"

    @torch.jit.script_method
    def get_name(self):
        return "LogisticRegression"


FLAGS = None


def main():
    lr = LogisticRegression(FLAGS.input_dim)
    lr.save("lr.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="data input dim."
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()

