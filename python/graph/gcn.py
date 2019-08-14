import os.path as osp

import torch
import torch.nn.functional as F

from nn.conv import GCNConv, GCNConv2  # noqa
from nn.conv import SAGEConv 



class GCN(torch.jit.ScriptModule):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    @torch.jit.script_method 
    def forward_(self, x, edge_index):
        # type: (Tensor, Tensor) -> Tensor
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    @torch.jit.script_method
    def loss(self, outputs, targets):
        return F.nll_loss(outputs, targets)
  
    def get_training(self):
        return self.training

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
    # gcn = GCN2(1433, 16. 7) # cora dataset
    # gcn = GCN2(23, 64, 7) # eth dataset
    # gcn = GCN2(602, 64, 41) # reddit dataset
    gcn = GCN2(233, 128, 2) # huge
    gcn.save('gcn2_huge.pt')
