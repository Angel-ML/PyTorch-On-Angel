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
    gcn = GCN2(1433, 16, 7)
    gcn.save('gcn2.pt')

# size = 0
# for (name, param) in model.named_parameters():
#     # print(name, param)
#     size += param.view(-1).size(0)

# print(size)


'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    # print(model.get_training())
    optimizer.zero_grad()
    train = data.train_mask
    output = model(data)
    F.nll_loss(output[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    # print(model.get_training())
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 200):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
'''