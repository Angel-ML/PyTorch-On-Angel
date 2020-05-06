# Tencent is pleased to support the open source community by making Angel available.
#
# Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/Apache-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
#
# !/usr/bin/env python

from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F


class LogisticRegression(torch.nn.Module):

    def __init__(self, input_dim=-1):
        super(LogisticRegression, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim

        # local model do not need real input_dim to init params, so set fake_dim to
        # speed up to produce local pt file.
        fake_input_dim = 10
        if input_dim > 0:
            self.bias = torch.zeros(1, 1)
            self.weights = torch.randn(fake_input_dim, 1)
            self.bias = torch.nn.Parameter(self.bias, requires_grad=True)
            self.weights = torch.nn.Parameter(self.weights, requires_grad=True)
            torch.nn.init.xavier_uniform_(self.weights)

    def forward_(self, batch_size, index, feats, values, bias, weight):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
        index = index.view(-1)
        values = values.view(1, -1)
        srcs = weight.view(1, -1).mul(values).view(-1)
        output = torch.zeros(batch_size, dtype=torch.float32, device="cuda")
        output.scatter_add_(0, index, srcs)
        output = output + bias
        return torch.sigmoid(output)

    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        weight = F.embedding(feats, self.weights)
        bias = self.bias
        return self.forward_(batch_size, index, feats, values, bias, weight)

    @torch.jit.export
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.export
    def get_type(self):
        return "BIAS_WEIGHT"

    @torch.jit.export
    def get_name(self):
        return "LogisticRegression"


FLAGS = None


def main():
    lr = LogisticRegression(FLAGS.input_dim)
    lr_script_module = torch.jit.script(lr)
    lr_script_module.save("lr.pt")


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
