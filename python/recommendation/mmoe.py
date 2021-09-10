from __future__ import print_function
import argparse

from torch import Tensor
from typing import List


import torch
import torch.nn.functional as F


class MMOE(torch.nn.Module):
    def __init__(self, input_dim, experts_num, experts_out, experts_hidden, towers_hidden, tasks):
        super(MMOE, self).__init__()
        # params
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim
        self.experts_num = experts_num
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks
        self.softmax = torch.nn.Softmax(dim=1)

        """input layers embedding"""
        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))
        # weights
        self.weights = torch.nn.Parameter(torch.zeros(1, 1))
        # embeddings
        # self.embedding = torch.nn.Parameter(torch.zeros(embedding_dim))

        # expert_weight
        self.w_expert = [torch.nn.Parameter(torch.zeros(input_dim, experts_hidden)),torch.nn.Parameter(torch.zeros(experts_hidden,experts_out))]*experts_num

        # gates_weight
        self.w_gates = [torch.nn.Parameter(torch.zeros(input_dim, experts_num))]*tasks

        # tower_weight
        self.w_towers = [torch.nn.Parameter(torch.zeros(experts_out, towers_hidden)),torch.nn.Parameter(torch.zeros(towers_hidden,1))]*tasks

        # mats
        self.mats = self.w_expert + self.w_gates+self.w_towers

        # init
        for i in self.mats:
            torch.nn.init.xavier_uniform_(i)


        """Angel Params"""
        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))
        # weights
        self.weights = torch.nn.Parameter(torch.zeros(1, 1))


    def forward_(self,batch_size, index, feats, values,mats):
        # type: (int, Tensor, Tensor, Tensor,List[Tensor]) -> Tensor
        index = index.view(-1)
        values = values.view(1, -1)

        # get the experts output
        w_expert = mats[0:self.experts_num*2]
        # print("======w_expert=======")
        # print(w_expert)
        expers_outs = []
        for w_expert0,w_expert1 in zip(w_expert[0::2], w_expert[1::2]):
            w_expert0 = F.embedding(feats, w_expert0)
            srcs = w_expert0.mul(values.view(-1,1)).transpose_(0,1)
            expert_out = torch.zeros(batch_size,self.experts_hidden, dtype=torch.float32)
            index_expert = index.repeat(self.experts_hidden).view(self.experts_hidden, -1)
            expert_out.scatter_add_(1, index_expert, srcs)
            expert_out = torch.relu(expert_out)
            expert_out = expert_out @ expert_out
            expers_outs.append(expert_out)

        expers_out_tensor = torch.stack(expers_outs)

        #get the gates output
        # w_gates =  mats.pop(self.tasks)
        w_gates = mats[self.experts_num*2:self.experts_num*2+self.tasks]
        srcs = [ F.embedding(feats,w_gate).mul(values.view(-1,1)).transpose_(0,1) for w_gate in w_gates]
        index_gate = index.repeat(self.experts_num).view(self.experts_num, -1)
        gates_out = torch.zeros(batch_size,self.experts_num, dtype=torch.float32)
        gates_outs = [gates_out.scatter_add_(0,index_gate,src) for src in srcs]
        gates_outs = [self.softmax(gates_out)for gates_out in gates_outs]

        #get the towers_input
        towers_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * expers_out_tensor for g in gates_out]
        towers_input = [torch.sum(ti, dim=0) for ti in towers_input]

        # get the final output from the towers
        w_towers = mats[self.experts_num*2+self.tasks:]
        final_output = []
        for w_tower0, w_tower1,i in zip(w_towers[0::2], w_towers[1::2],range(self.tasks)):
            srcs = w_tower0.view(1,-1).mul(towers_input[i].view(1, -1)).view(-1)
            tower_out = torch.zeros(batch_size, dtype=torch.float32)
            tower_out.scatter_add_(0, index, srcs)
            tower_out = torch.relu(tower_out)
            srcs = w_tower1.view(1,-1).mul(tower_out).view(-1)
            tower_out.scatter_add_(0, index, srcs)
            final_output.append(tower_out)

        # get the output of the towers, and stack them
        final_output = torch.stack(final_output, dim=1)
        return final_output

    def forward(self, batch_size: int, index, feats, values):
        return self.forward_(batch_size, index, feats, values, self.mats)


    @torch.jit.export
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.export
    def get_type(self):
        return "BIAS_WEIGHT_EMBEDDING_MATS"

    @torch.jit.export
    def get_name(self):
        return "mmoe"

def main():
    mmoe = MMOE(FLAGS.input_dim, FLAGS.experts_num, FLAGS.experts_out, FLAGS.experts_hidden, FLAGS.towers_hidden, FLAGS.tasks)
    # mmoe = MMOE(input_dim=5, experts_num=3, experts_out=4, experts_hidden=2, towers_hidden=2, tasks=2)
    mmoe_script_module = torch.jit.script(mmoe)
    mmoe_script_module.save("mmoe.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="data input dim."
    )
    parser.add_argument(
        "--experts_num",
        type=int,
        default=-1,
        help="experts num"
    )
    parser.add_argument(
        "--experts_out",
        type=int,
        default=-1,
        help="experts out dim"
    )
    parser.add_argument(
        "--experts_hidden",
        type=int,
        default=-1,
        help="experts hidden dim"
    )
    parser.add_argument(
        "--towers_hidden",
        type=int,
        default=-1,
        help="towers hidden dim"
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=-1,
        help="tasks num"
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()

# python mmoe.py --input_dim 5 --experts_num 3 --experts_out 4 --experts_hidden 2 --towers_hidden 2  --tasks 2
