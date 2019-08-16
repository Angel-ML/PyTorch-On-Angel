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
import os.path as osp

import torch
import torch.nn.functional as F

from nn.conv import GCNConv2  # noqa


class GCN2(torch.jit.ScriptModule):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv2(input_dim, hidden_dim)
        self.conv2 = GCNConv2(hidden_dim, num_classes)

    @torch.jit.script_method 
    def forward_(self, x, first_edge_index, second_edge_index):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        x = F.relu(self.conv1(x, second_edge_index))
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, first_edge_index)
        return F.log_softmax(x, dim=1)

    @torch.jit.script_method
    def loss(self, outputs, targets):
        return F.nll_loss(outputs, targets)
  
    def get_training(self):
        return self.training

if __name__ == '__main__':
    gcn = GCN2(1433, 16. 7) # cora dataset
    gcn.save('gcn.pt')