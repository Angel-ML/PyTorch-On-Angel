import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

import torch_geometric.transforms as T

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

from gcn import GCN
from sage import SAGE

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


num_nodes = data.x.size(0)
input_dim = data.x.size(1)
hidden_dim = 16
num_classes = 7

model = GCN(input_dim, hidden_dim, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0)

for epoch in range(1, 2000):
    optimizer.zero_grad()
    output = model.forward_(data.x, data.edge_index)
    loss = model.loss(output, data.y)
    loss.backward()
    optimizer.step()
    acc = output.max(1)[1].eq(data.y).sum().item() / num_nodes
    print('epoch=%d loss=%f acc=%f' % (epoch, loss.item(), acc))