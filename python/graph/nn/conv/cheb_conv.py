import torch
from torch.nn import Parameter

from utils import scatter_add
from utils import remove_self_loops
from utils import spmm
from utils import uniform

class ChebConv(torch.jit.ScriptModule):
    def __init__(self, in_channels, out_channels, K, bias=True):
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    @torch.jit.script_method
    def forward(self, x, edge_index, edge_weight=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index[0], edge_index[1]
        num_nodes, num_edges, K = x.size(0), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = torch.ones((num_edges, ),
                                     dtype=x.dtype, 
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        # Compute normalized and rescaled graph Laplacian
        deg = deg.pow(-0.5)
        lap = -deg[row] *edge_weight * deg[col]

        # Perform filter operation recurrently
        Tx_0 = x
        out = torch.mm(Tx_0, self.weight[0])
        Tx_1 = spmm(edge_index, lap, num_nodes, x)

        if K > 1:
            out = out + torch.mm(Tx_1, self.weight[1])
        
        for k in range(K):
            if k >= 2:
                Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0
                out = out + torch.mm(Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out
    
    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))