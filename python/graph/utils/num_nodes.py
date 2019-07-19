import torch

@torch.jit.script
def maybe_num_nodes(index, num_nodes=None):
	# type: (Tensor, Optional[int]) -> int
    return int(index.max().item()) + 1 if num_nodes is None else num_nodes