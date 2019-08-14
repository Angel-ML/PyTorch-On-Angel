import os.path as osp

# import sys
# path = '/Users/leleyu/workspace/github/geometric/pytorch_geometric/build/lib'
# sys.path = [path] + sys.path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DeepGraphInfomax, GCNConv
from sklearn.linear_model import LogisticRegression
import numpy as np
# from gcn import GCN
# from sage import SAGE
from dgi import DGI

import torch_geometric.transforms as T

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
# dataset = Planetoid(path, dataset, T.NormalizeFeatures())
dataset = Planetoid(path, dataset)
data = dataset[0]

num_nodes = data.x.size(0)
input_dim = data.x.size(1)
hidden_dim = 512
num_classes = 7

model = DGI(input_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

for epoch in range(1, 300):
    neg_x = data.x[torch.randperm(num_nodes)]
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model.forward_(pos_x=data.x, neg_x=neg_x, edge_index=data.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    
    print("Epoch: {:3d}, Loss: {:.4f}".format(epoch, loss.item()))
    # acc = output.max(1)[1].eq(data.y).sum().item() / num_nodes
    # print('epoch=%d loss=%f acc=%f' % (epoch, loss.item(), acc))


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DeepGraphInfomax(
#     hidden_channels=hidden_dim, encoder=Encoder(dataset.num_features, hidden_dim),
#     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
#     corruption=corruption).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def make_minibatch_idx(edge_index, batch_size):
    start = 0
    res = []
    allSrcs = list(set(edge_index[0].tolist()))
    for srcs in [allSrcs[i : min(i + batch_size, edge_index.shape[1])] for i in range(0, edge_index.shape[1], batch_size)]:
        batch_ei = None
        for s in srcs:
            batch_ei = edge_index[:, edge_index[0] == s] if batch_ei is None else \
                torch.cat((batch_ei, edge_index[:, edge_index[0] == s]), 1)
        if batch_ei is not None:
            res.append(batch_ei)
    return res

batch_edge_index = make_minibatch_idx(data.edge_index, 500)

def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(data.x, data.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()

def minibatch_train():
    totalLoss = 0
    for edge_index in batch_edge_index:
        model.train()   
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x, edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
    return totalLoss

def test():
    model.eval()
    # z, _, _ = model(data.x, data.edge_index)
    z = model.predict_(data.x, data.edge_index)
    clf = LogisticRegression(
        solver='lbfgs', multi_class='auto', max_iter=150).fit(
            z[data.train_mask].detach().cpu().numpy(), data.y[data.train_mask].detach().cpu().numpy())
    train_score = clf.score(z[data.train_mask].detach().numpy(), data.y[data.train_mask].detach().numpy())
    test_score =  clf.score(z[data.test_mask].detach().numpy(), data.y[data.test_mask].detach().numpy())
    return train_score, test_score

# for epoch in range(1, 301):
#     loss = train()
#     print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
train_acc, test_acc = test()
print('Train Accuracy: {:.4f} Test Accuracy: {:.4f}'.format(train_acc, test_acc))