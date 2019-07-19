#!/usr/bin/env python

import torch
import torch.nn.functional as F
from nn.conv import SAGEConv, SAGEConv2

class SAGE(torch.jit.ScriptModule):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)

    @torch.jit.script_method
    def forward_(self, x, edge_index):
        # type: (Tensor, Tensor) -> Tensor
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    @torch.jit.script_method
    def loss(self, outputs, targets):
        return F.nll_loss(outputs, targets)



class SAGE2(torch.jit.ScriptModule):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SAGE2, self).__init__()
        self.conv1 = SAGEConv2(input_dim, hidden_dim)
        self.conv2 = SAGEConv2(hidden_dim, num_classes)

    @torch.jit.script_method
    def forward_(self, x, first_edge_index, second_edge_index):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        # print(x.size())
        x = F.relu(self.conv1(x, second_edge_index))
        # print(x.size())
        x = self.conv2(x, first_edge_index)
        # print(x.size())
        return F.log_softmax(x, dim=1)

    @torch.jit.script_method
    def loss(self, outputs, targets):
        return F.nll_loss(outputs, targets)

if __name__ == '__main__':
    sage = SAGE2(1433, 16, 7)
    sage.save('sage2.pt')