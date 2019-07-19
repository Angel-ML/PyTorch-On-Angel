import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

# print(data.x)
# print(data.edge_index)
# print(data.y)
# print(data.train_mask)
# y = data.y.numpy()
# labels = dict()
# for (idx, label) in enumerate(y):
#     labels[idx] = label
    # writer.write('%s %s\n' % (str(idx), str(label)))

# x = data.x.numpy()
# writer = open('cora_label_feature', 'w')
# for (idx, f) in enumerate(x):
#     ff = []
#     for (j, v) in enumerate(f):
#         if v:
#             ff.append('%s:%s' % (str(j), str(v)))
#     if len(ff) > 0:
#         writer.write('%s %s %s\n' % (str(idx), str(labels[idx]), ' '.join(ff)))

edge_index = data.edge_index.numpy()
src = edge_index[0]
dst = edge_index[1]
print(len(set(src)))
print(len(set(dst)))