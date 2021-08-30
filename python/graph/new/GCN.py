import os.path as osp
import os
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)#全连接层

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)#2708 X 16

        # Step 3: Compute normalization.
        row, col = edge_index

        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)
#self.message中对邻居节点作linear变换，然后归一化
#self.aggregate以sum方式聚合
#self.update对聚合后特征不作处理

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
#卷积运算定义

class Net(torch.nn.Module):
    # torch.nn.Module 是所有神经网络单元的基类
    def __init__(self):
        super(Net, self).__init__()  ###复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)#定义图网络的两层卷积,中间层为16

    def forward(self, data):
        x, edge_index = data.x, data.edge_index#2708 X 1433 , 2 X 10556 
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


##############################设置GPU、定义优化器#############################
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath('__file__')), 'data', dataset)
## 加载数据集
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, data = Net().to(device), dataset[0].to(device)
print(data)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


#网络训练
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)#调用forward前向传播
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])#对多分类任务计算loss
    loss.backward()#反向传播计算梯度
    optimizer.step()#利用优化器优化

#测试
model.eval()
out = model(data)
_, pred = model(data).max(dim=1)#预测概率最高的作为预测结果,第一个返回是value，第二个返回是classs
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))