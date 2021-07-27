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


@torch.jit.script
def parse_feat(x, batch_ids, field_ids, field_num, encode):
    # type: (Tensor, Tensor, Tensor, int, str) -> Tensor
    k = x.size(1)# embedding_dim
    if encode == "multi-hot":
        b = torch._unique(batch_ids, sorted=False)[0].size(0)# batchsize
        f = field_num
        t_index = [batch_ids.view(-1).to(torch.long), field_ids.view(-1).to(torch.long)]
        e_transpose = x.view(-1, k).transpose(0, 1)
        count = torch.ones(x.size(0))

        hs = []
        for i in range(k):
            h = torch.zeros(b, f)
            c = torch.zeros(b, f)
            h.index_put_(t_index, e_transpose[i], True)
            c.index_put_(t_index, count, True)  # sum
            h = h / c.clamp(min=1)  # avg
            hs.append(h.view(-1, 1))

        emb_cat = torch.cat(hs, dim=1)
        output = emb_cat.view(b, -1)

    elif encode == "one-hot":
        output = x.view(-1, field_num * k)
    else:
        output = x

    return output
