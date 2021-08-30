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
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

import torch_geometric.transforms as T
import numpy as np
import random
from time import perf_counter


def train_gcn():
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath('__file__')), 'data', dataset)
    ## 加载数据集
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    from gcn import GCN

    num_nodes = data.x.size(0)
    input_dim = data.x.size(1)
    hidden_dim = 16
    num_classes = 7
    model = GCN(input_dim, hidden_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0)
    t = perf_counter()
    for epoch in range(1, 2000):
        optimizer.zero_grad()
        output = model.forward_(data.x, data.edge_index)
        loss = model.loss(output, data.y)
        loss.backward()
        optimizer.step()
        acc = output.max(1)[1].eq(data.y).sum().item() / num_nodes
        print('epoch=%d loss=%f acc=%f' % (epoch, loss.item(), acc))
    train_time = perf_counter()-t
    print(train_time)

def train_sgcn():
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath('__file__')), 'data', dataset)
    ## 加载数据集
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    from sgcn import SGCN

    num_nodes = data.x.size(0)
    input_dim = data.x.size(1)
    num_classes = 7

    model = SGCN(input_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0)
    t = perf_counter()
    for epoch in range(1, 2000):
        optimizer.zero_grad()
        output = model.forward_(data.x, data.edge_index)
        loss = model.loss(output, data.y)
        loss.backward()
        optimizer.step()
        acc = output.max(1)[1].eq(data.y).sum().item() / num_nodes
        print('epoch=%d loss=%f acc=%f' % (epoch, loss.item(), acc))
    train_time = perf_counter()-t
    print(train_time)

def mix_train_test(name, test_ratio):
    from torch_geometric.datasets import Entities
    path = osp.join(
        osp.dirname(osp.realpath(__file__)), './', 'data', 'Entities', name)
    dataset = Entities(path, name)
    data = dataset[0]

    train_test_idx = torch.cat([data.train_idx, data.test_idx])
    train_test_y = torch.cat([data.train_y, data.test_y])

    size = train_test_idx.size(0)
    index = np.array([x for x in range(size)])
    random.shuffle(index)
    index = torch.from_numpy(index)
    train_size = int(size * (1 - test_ratio))
    train_idx, train_y = train_test_idx[0:train_size], train_test_y[0:train_size]
    test_idx, test_y = train_test_idx[train_size:], train_test_y[train_size:]
    return train_idx, test_idx, train_y, test_y


def train_rgcn():
    from torch_geometric.datasets import Entities
    name = 'MUTAG'
    path = osp.join(
        osp.dirname(osp.realpath(__file__)), './', 'data', 'Entities', name)
    dataset = Entities(path, name)#下载数据集到path
    data = dataset[0]
    print(data)
    os.system("pause")
    from rgcn import RGCN
    x = torch.zeros(data.num_nodes, 16)
    torch.nn.init.xavier_uniform_(x)#对x进行初始化

    model = RGCN(x.size(1), 16, dataset.num_relations, 30, dataset.num_classes)#创建模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)#定义优化器内容

    train_idx, test_idx, train_y, test_y = mix_train_test(name, 0.2)#分离训练集和测试集
    print('train_size: {:03d}, test_size: {:03d}'.format(train_idx.size(0), test_idx.size(0)))

    def train():
        model.train()
        optimizer.zero_grad()
        out = model.forward_(x, data.edge_index, data.edge_type, None)
        loss = F.nll_loss(out[train_idx], train_y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def test():
        model.eval()
        out = model.forward_(x, data.edge_index, data.edge_type, None)
        pred = out[test_idx].max(1)[1]
        test_acc = pred.eq(test_y).sum().item() / test_y.size(0)
        pred = out[train_idx].max(1)[1]
        train_acc = pred.eq(train_y).sum().item() / train_y.size(0)
        return train_acc, test_acc
    

    for epoch in range(1, 100):
        loss = train()
        train_acc, test_acc = test()
        print('Epoch: {:02d}, Loss: {:.4f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}'
              .format(epoch, loss, train_acc, test_acc))


def download_data():
    from torch_geometric.datasets import Entities
    name = 'AM'
    path = osp.join(
        osp.dirname(osp.realpath(__file__)), './', 'data', 'Entities', name)
    dataset = Entities(path, name)
    data = dataset[0]

    print(dataset.num_relations)
    print(dataset.num_classes)
    print(data.num_nodes)
    print(data.train_idx.size(0))
    print(data.test_idx.size(0))

    # write edge_index and edge_types

    writer = open('data/Entities/AM/am_edge', 'w')
    src, dst = data.edge_index[0], data.edge_index[1]
    size = src.size(0)
    for idx in range(size):
        writer.write('%d %d %s\n' % (src[idx].item(), dst[idx].item(),
                                     data.edge_type[idx].item()))

    # write labels
    writer = open('data/Entities/AM/am_label', 'w')
    train_idx, train_y = data.train_idx, data.train_y
    size = train_idx.size(0)
    for idx in range(size):
        writer.write('%d %d\n' % (train_idx[idx].item(), train_y[idx].item()))

    test_idx, test_y = data.test_idx, data.test_y
    size = test_idx.size(0)
    for idx in range(size):
        writer.write('%d %d\n' % (test_idx[idx].item(), test_y[idx].item()))

    x = torch.zeros(data.num_nodes, 32)
    torch.nn.init.xavier_uniform_(x)
    size = data.num_nodes
    writer = open('data/Entities/AM/am_feature', 'w')
    for idx in range(size):
        fs = x[idx].numpy()
        fs = [str(f) for f in fs]
        writer.write('%d\t%s\n' % (idx, ' '.join(fs)))


if __name__ == '__main__':
    # mix_train_test()
    train_gcn()
    #train_sgcn()
    # download_data()