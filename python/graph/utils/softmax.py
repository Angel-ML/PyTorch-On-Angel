import torch
from .num_nodes import maybe_num_nodes
from utils import scatter_add, scatter_max

@torch.jit.script
def softmax(src, index, num_nodes):
    # type: (Tensor, Tensor, int) -> Tensor
    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[index]
    out = out.exp() / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out