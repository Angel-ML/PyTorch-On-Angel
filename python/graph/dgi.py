import torch
import torch.nn.functional as F
from torch.nn import Parameter
from nn.conv import GCNConv, SAGEConv3, GCNConv2, SAGEConv4

class DGI(torch.jit.ScriptModule):
    def __init__(self, n_in, n_h):
        super(DGI, self).__init__()
        self.n_h = n_h
        self.gcn = SAGEConv3(n_in, n_h)
        self.weight = Parameter(torch.Tensor(n_h, n_h))
        self.reluWeight = Parameter(torch.Tensor(n_h).fill_(0.25))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    
    @torch.jit.script_method
    def forward_(self, pos_x, neg_x, edge_index):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = F.prelu(self.gcn(pos_x, edge_index), self.reluWeight)
        neg_z = F.prelu(self.gcn(neg_x, edge_index), self.reluWeight)
        summary = torch.sigmoid(torch.mean(pos_z, dim=0))
        return pos_z, neg_z, summary
    
    @torch.jit.script_method
    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()

        return pos_loss + neg_loss

    @torch.jit.script_method
    def discriminate(self, z, summary, sigmoid=True):
        # type: (Tensor, Tensor, bool) -> Tensor
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value
 
    @torch.jit.script_method
    def predict_(self, x, edge_index):
        return F.prelu(self.gcn(x, edge_index), self.reluWeight)
        
if __name__ == '__main__':
    dgi = DGI(1433, 512)
    dgi.save('dgi_gcn.pt')