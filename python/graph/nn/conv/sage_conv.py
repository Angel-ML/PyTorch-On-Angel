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
import torch
import torch.nn.functional as F
from torch.nn import Parameter


from utils import uniform, glorot, zeros
from utils import add_self_loops, remove_self_loops
from utils import scatter_mean

class SAGEConv3(torch.jit.ScriptModule):

    '''
    The difference between SAGEConv2 and SAGEConv is that SAGEConv2 is designed for 
    distributed mini-batch training. The key point is that we do not add self-loops
    for edge_index in SAGEConv2 since we add loops before passing edge_index into pytorch.
    The reason that why we add loops outside pytorch is that if we only need add loops for 
    the inner-nodes and first-order nodes. Thus, we can reduce some calculation.
    '''
    def __init__(self, in_channels, out_channels):
        super(SAGEConv3, self).__init__()

        self.weight = Parameter(torch.Tensor(in_channels * 2, out_channels))
        self.bias = Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias.numel() > 1:
            uniform(self.bias.numel(), self.bias)

    @torch.jit.script_method
    def forward(self, x, edge_index):
        # type: (Tensor, Tensor) -> Tensor
        row, col = edge_index[0], edge_index[1]
        out = scatter_mean(x[col], row, dim=0) # do not set dim_size, out.size() = row.max() + 1
        x = x[0:out.size(0)]
        x = torch.cat([x, out], dim=1)
        out = torch.matmul(x, self.weight)
        out = out + self.bias
        out = F.normalize(out, p=2.0, dim=-1)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
            self.out_channels)


