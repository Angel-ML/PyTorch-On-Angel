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

import torch
import torch.nn.functional as F

from nn.conv import SGCNConv, SGCNConv2  # noqa


class SGCN(torch.jit.ScriptModule):
    def __init__(self, input_dim, num_classes):
        super(SGCN, self).__init__()
        self.conv1 = SGCNConv(input_dim, num_classes)

    @torch.jit.script_method
    def forward_(self, x, edge_index):
        # type: (Tensor, Tensor) -> Tensor
        x = self.conv1(x, edge_index)
        # x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)

    @torch.jit.script_method
    def loss(self, outputs, targets):
        return F.nll_loss(outputs, targets)

    def get_training(self):
        return self.training


class SGCN2(torch.jit.ScriptModule):
    def __init__(self, input_dim, num_classes):
        super(SGCN2, self).__init__()
        self.conv1 = SGCNConv2(input_dim, num_classes)

    @torch.jit.script_method
    def forward_(self, x, first_edge_index, second_edge_index):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        x = self.conv1(x, second_edge_index)
        # x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)

    @torch.jit.script_method
    def loss(self, outputs, targets):
        return F.nll_loss(outputs, targets)

    def get_training(self):
        return self.training


if __name__ == '__main__':
    # gcn = GCN2(1433, 16. 7) # cora dataset
    # gcn = GCN2(23, 64, 7) # eth dataset
    # gcn = GCN2(602, 64, 41) # reddit dataset
    sgcn = SGCN2(1433, 7)  # huge
    sgcn.save('sgcn.pt')