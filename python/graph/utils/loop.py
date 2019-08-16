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
import torch

from .num_nodes import maybe_num_nodes


@torch.jit.script
def contains_self_loops(edge_index):
    # type: (Tensor) -> bool
    row, col = edge_index[0], edge_index[1]
    mask = row == col
    return int(mask.sum()) > 0

@torch.jit.script
def remove_self_loops(edge_index, edge_attr=None):
    # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr

@torch.jit.script
def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    # type: (Tensor, Optional[Tensor], int, Optional[int]) -> Tuple[Tensor, Optional[Tensor]]
    
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0,
                              num_nodes,
                              dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)


    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = torch.empty((num_nodes))
        loop_weight.fill_(fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


@torch.jit.script
def add_remaining_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=None):
    # type: (Tensor, Tensor, int, Optional[int]) -> Tuple[Tensor, Tensor]

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]

    mask = row != col
    inv_mask = 1 - mask
    loop_weight = torch.full(
        (num_nodes, ),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    
    assert edge_weight.numel() == edge_index.size(1)
    loop_weight[row[inv_mask]] = edge_weight[inv_mask].view(-1)
    edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    loop_index = torch.arange(0,
        num_nodes,
        dtype=torch.long,
        device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight
