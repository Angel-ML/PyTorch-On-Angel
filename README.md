## SGCN
GCNs 的灵感主要来自最近的深度学习方法可能会继承不必要的复杂性和冗余计算。在本文中，作者减少了这种多余的复杂性，并通过实验证明了该方法可以有效提高网络的计算速度且不会对精度产生影响。S-GCN相比于 Faster-GCN 可以把计算速度提高两个数量级。作者通过反复去除 GCN 层之间的非线性，并将生成的函数折叠成单个线性变换，来减少 GCN 的过度复杂性。实验表明，最终的线性模型在各种任务上表现出与 GCNs 相当甚至更好的性能，同时计算效率更高，拟合的参数明显更少。这个简化后的模型称为 S-GCN。
### 取消隐藏层激活函数以消除非线性
以SGCN作者给出代码为例：
```python
class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)
#SGC每次只有一个全连接网络
    def forward(self, x):
        return self.W(x)
```
这里前向传播网络仅仅只是一个全连接网络，没有隐藏层和激活函数。
论文原文对这段叙述为，图卷积的好处大部分来自于局部平均还不是隐藏层中的非线性变化。原SGC公式简化为：
![image](https://user-images.githubusercontent.com/39088547/131299217-fad8cf7c-8143-499b-bdf7-e3da92f1e6ed.png)

这里X为原始输入，theta可以理解成节点神经元参数，S理解成边的参数（邻接矩阵）。

发现这里S中是没有需要训练的参数的，而theta只需要一个全连接层即可。
因此，作者删除原GCN模型中的隐藏层部分，改为一个输入为特征向量大小，输出为类别数的全连接层，但注意数据在输入SGN前需要增加预处理计算部分。

### 增加预处理计算
原论文作者提供代码下，实际train过程如下：
```python
model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
print("{:.4f}s".format(precompute_time))

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        print(output)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t
```
这里主要任务就是为sgcn添加sgc_precompute函数,函数内容如下：
```python
def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time
```
论文中采用默认degree为2，因此我们实验中也预设degree=2，实际如果需要的话这里可以设一个变量处理。  
sgc的输入为邻接矩阵，但我们原论文中的GCN_conv并不是采用邻接矩阵实现的，原论文实现图卷积代码如下：
```python
  def forward(self, x, edge_index):
        # type: (Tensor, Tensor) -> Tensor
        x = torch.matmul(x, self.weight)
        print(x.shape)
        edge_index, norm = self.norm(edge_index, x.size(0))

        return self.propagate(edge_index, x=x, norm=norm)
```
这里的self.weight相当于上面定义的全连接网络参数
```python
    def norm(self, edge_index, num_nodes):
        # type: (Tensor, int) -> Tuple[Tensor, Tensor]
        edge_weight = torch.ones((edge_index.size(1),),
                                 device=edge_index.device)

        fill_value = 1
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        print(edge_index.shape,edge_weight.shape)#13264条边，其中有2708条自循环边
        row, col = edge_index[0], edge_index[1]
        print(row.shape)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)#deg 大小为2708
        print(deg.shape)
        deg_inv_sqrt = deg.pow(-0.5)

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
```
输入数据送入propagate之前需要先经过norm函数处理：
这里对图卷积的实现参考：
![image](https://user-images.githubusercontent.com/39088547/131299684-931cf597-2d8e-44aa-8c24-fb925a7b0cf4.png)

和邻接矩阵实现对比：

![image](https://user-images.githubusercontent.com/39088547/131299712-59daacb3-8417-491f-945b-f79ef217593d.png)

因为添加了自循环和归一化，在特征传递效果上理论上是更好的（这句胡说八道的QAQ）
Norm返回从上面公式看就相当于邻接矩阵，两者效果相同，最后输出如下：
```python
    def propagate(self, edge_index, x, norm):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        x_j = torch.index_select(x, 0, edge_index[1]) #对输入数据按边的dst节点进行索引，有多少边，索引出来多少节点；有重复节点
        out = self.message(x_j, norm) #特征沿边传递
        out = scatter_add(out, edge_index[0], 0, None, dim_size=x.size(0)) #对所有边的传递结果进行汇总
        out = self.update(out)
        return out #输出最后结果

```
综上对GCNconv进行简单修改写一个SGCNconv
```python
    @torch.jit.script_method
    def propagate(self, edge_index, x, norm):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        x_j = torch.index_select(x, 0, edge_index[1])#对输入数据按边的dst节点进行索引，有多少边，索引出来多少节点，有重复节点
        out = self.message(x_j, norm)
        out = self.message(out,norm)#这里做两次邻接矩阵传递，后期可以添加degree遍历控制传递次数
        out = scatter_add(out, edge_index[0], 0, None, dim_size=x.size(0))
        out = self.update(out)
        return out
```
然后在模型其他地方简单修改即可。
### 实验结果
我们主要考虑训练速度的问题，采用perf_counter记录时间；

对sgcn做2000次训练：  
  ![image](https://user-images.githubusercontent.com/39088547/131299918-6126d450-28ff-4cfc-b414-e1c0ed3d5ce0.png)

对两层gcn做2000次训练：  
  ![image](https://user-images.githubusercontent.com/39088547/131299953-22e94908-15c9-4680-9af3-ccfa4ba7a29c.png)

问题：sgcn参数初始化精度非常差，原文作者程序里应该是采用了如下方法解决：
```python
    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)
```

但我们需要在输入数据前创建模型，这部分并不是特别好改。
可以看出sgcn训练时间明显快很多，但初始命中率低可能需要适当调大步长来加快收敛速度，另外sgcn训练稳定时候的命中率多数情况下是低于gcn的，所以比较合适的解决方案可能是预设一个较大的步长，然后让步长在训练过程中慢慢衰减。关于损失函数因为原论文也没有对这方面说的很清楚所以这块仍然保留了GCN的损失。

-by 沈志。

