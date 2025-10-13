#!/usr/bin/env python

from __future__ import print_function
import sys
sys.path.extend(["./", "../../"])

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math
from utils import parse_feat
import sys


class GATNEModel(torch.jit.ScriptModule):
    def __init__(
            self, in_dims, input_embedding_dims, input_field_nums, input_split_idxs, embedding_dim, embedding_u_dim,
            dim_a, node_types, edge_types, schema, aggregation_type, encode, dropout):
        super(GATNEModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding_u_dim = embedding_u_dim
        self.aggregation_type = aggregation_type
        self.get_node_types = node_types
        self.get_edge_types = edge_types
        self.get_schema = schema
        self.get_field_nums = input_field_nums
        self.get_feat_embed_dims = input_embedding_dims
        self.get_feature_dims = in_dims
        self.get_input_split_idxs = input_split_idxs
        self.encode = encode
        self.hidden_dropout = torch.nn.Dropout(dropout)

        self.input_dims = in_dims.split(",")
        self.embedding_dims = input_embedding_dims.split(",")
        self.field_nums = input_field_nums.split(",")
        self.split_idxs = input_split_idxs.split(",")
        self.node_types = [int(i) for i in node_types.split(",")]
        self.edge_types = [int(i) for i in edge_types.split(",")]
        self.edge_type_count = len(self.edge_types)
        self.schema = {}
        for h_r_t in schema.split(","):
            h, r, t = h_r_t.split("-")
            self.schema[h+'_'+r] = int(t)
            self.schema[t+'_'+r] = int(h)

        self.input_dims_idx = {}
        self.embedding_dims_idx = {}
        self.field_nums_idx = {}
        self.node_split_idxs = {}

        # parse input dims, id:dim
        for _, pair in enumerate(self.input_dims):
            kv = pair.split(":")
            self.input_dims_idx[int(kv[0])] = int(kv[1])
            self.field_nums_idx[int(kv[0])] = -1
            self.node_split_idxs[int(kv[0])] = 0

        # parse input embedding dims, id:dim
        if len(input_embedding_dims) > 0:
            for _, pair in enumerate(self.embedding_dims):
                kv = pair.split(":")
                self.embedding_dims_idx[int(kv[0])] = int(kv[1])

        # parse input field_nums, id:field_num
        if len(input_field_nums) > 0:
            for _, pair in enumerate(self.field_nums):
                kv = pair.split(":")
                self.field_nums_idx[int(kv[0])] = int(kv[1])

        if len(input_split_idxs) > 0:
            for _, pair in enumerate(self.split_idxs):
                kv = pair.split(":")
                self.node_split_idxs[int(kv[0])] = int(kv[1])

        self.weights = torch.nn.ModuleList()
        for i in self.node_types:
            in_dim = self.input_dims_idx[i]
            if len(input_embedding_dims) > 0:
                emb_dim = self.embedding_dims_idx[i]
                field_num = self.field_nums_idx[i]
                split_idx = self.node_split_idxs[i]
                in_dim = emb_dim * field_num + split_idx if emb_dim > 0 else in_dim

            self.weights.append(torch.nn.Linear(in_dim, self.embedding_dim, bias=False))
            for j in self.edge_types:
                self.weights.append(torch.nn.Linear(in_dim, self.embedding_u_dim, bias=False))

        self.trans_weights = Parameter(torch.FloatTensor(self.edge_type_count, self.embedding_u_dim, self.embedding_dim))
        self.trans_weights_s1 = Parameter(torch.FloatTensor(self.edge_type_count, self.embedding_u_dim, dim_a))
        self.trans_weights_s2 = Parameter(torch.FloatTensor(self.edge_type_count, dim_a, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.weights)):
            torch.nn.init.normal_(self.weights[i].weight, std=1.0 / math.sqrt(self.embedding_dim))

        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_dim))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_dim))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_dim))

    @torch.jit.script_method
    def embedding_(self, x, srcs, src_type, neighbors, neighbors_type, neighbors_flag, first_edge_type,
                   batch_ids=None, field_ids=None, x_dense=None):
        # type: (List[Tensor], Tensor, int, List[Tensor], Tensor, Tensor, Tensor, Optional[List[Tensor]], Optional[List[Tensor]], Optional[List[Tensor]]) -> Tensor

        # parse high-sparse feature
        feats = self.encode_feat(x, batch_ids, field_ids, x_dense)

        src_weight: torch.Tensor = torch.empty(0)
        dim_weight: torch.Tensor = torch.empty(0)

        for i, module in enumerate(self.weights):
            if i == src_type*(self.edge_type_count+1):
                src_weight = module.weight
        node_embed = torch.matmul(feats[src_type][srcs], torch.t(src_weight))       # batch, embedding_dim

        node_type_embed = []
        for edge_type in self.edge_types:
            edge_type_neigh = neighbors[edge_type]      # [batch_size, 10]
            flag = neighbors_flag[edge_type]            # [1]
            each_neighbors_type = neighbors_type[edge_type]     # [batch_size]

            if flag.item() == 1:
                neigh_type = neighbors_type[edge_type][0]
                for j, nei_module in enumerate(self.weights):
                    if j == neigh_type*(self.edge_type_count+1) + edge_type + 1:
                        dim_weight = nei_module.weight
                print("neighbor type: ", neigh_type, edge_type_neigh.shape, dim_weight.shape, feats[neigh_type][edge_type_neigh].shape, len(feats))
                edge_type_neigh_embedding = torch.matmul(feats[neigh_type][edge_type_neigh], torch.t(dim_weight))
            else:
                max_node_num = 0
                for feat in feats:
                    if feat.shape[0] > max_node_num:
                        max_node_num = feat.shape[0]

                new_feats = []              # [node_type, max_node_num, embedding_u_dim]
                for i, feat in enumerate(feats):
                    for k, nei_module in enumerate(self.weights):
                        if k == i*(self.edge_type_count+1) + edge_type + 1:
                            dim_weight = nei_module.weight
                    new_feat = torch.matmul(feat, torch.t(dim_weight))           # [each_node_num, embedding_u_dim]
                    cur_node_num = feat.shape[0]
                    new_feats.append(F.pad(new_feat, (0, 0, 0, max_node_num-cur_node_num), value=1.0))
                new_feats = torch.stack(new_feats, 0)
                neigh_type = each_neighbors_type.view(-1, 1)

                edge_type_neigh_embedding = new_feats[neigh_type, edge_type_neigh]

            if self.aggregation_type == 'sum':
                edge_type_embedding = torch.sum(edge_type_neigh_embedding, dim=1)
            elif self.aggregation_type == 'mean':
                edge_type_embedding = torch.mean(edge_type_neigh_embedding, dim=1)
            else:
                print('unsupported aggregation type')
                edge_type_embedding = torch.ones((len(srcs), self.embedding_u_dim))
            node_type_embed.append(edge_type_embedding)
        node_type_embed = torch.stack(node_type_embed, 1)       # batch, edge_type_count, embedding_u_dim

        trans_w = self.trans_weights[first_edge_type]           # embedding_u_dim, embedding_dim
        trans_w_s1 = self.trans_weights_s1[first_edge_type]     # embedding_u_dim, dim_a
        trans_w_s2 = self.trans_weights_s2[first_edge_type]     # dim_a, 1

        attention = F.softmax(
            torch.matmul(
                torch.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)                                          # batch, 1, edge_type_count
        node_type_embed = torch.matmul(attention, node_type_embed)  # batch, 1, embedding_u_dim
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze(1)     # batch, embedding_dim

        last_node_embed = F.normalize(node_embed, dim=1)

        return last_node_embed

    @torch.jit.script_method
    def forward_(self, x, srcs, src_type, context_x, negatives, neighbors, neighbors_type, neighbors_flag,
                 first_edge_type, batch_ids=None, field_ids=None, x_dense=None):
        # type: (List[Tensor], Tensor, int, Tensor, Tensor, List[Tensor], Tensor, Tensor, Tensor, Optional[List[Tensor]], Optional[List[Tensor]], Optional[List[Tensor]]) -> Tuple[Tensor, Tensor, Tensor]
        last_node_embed = self.embedding_(x, srcs, src_type, neighbors, neighbors_type, neighbors_flag, first_edge_type, batch_ids, field_ids, x_dense)
        return last_node_embed, context_x, negatives

    @torch.jit.script_method
    def encode_feat(self, x, batch_ids, field_ids, x_dense):
        # type: (List[Tensor], Optional[List[Tensor]], Optional[List[Tensor]], Optional[List[Tensor]]) -> List[Tensor]
        feats = []
        for nid, f in enumerate(x):
            if self.field_nums_idx[nid] > 0 and batch_ids is not None and field_ids is not None:
                feat = parse_feat(f, batch_ids[nid], field_ids[nid], self.field_nums_idx[nid], self.encode)
                if self.node_split_idxs[nid] > 0 and x_dense is not None:
                    feat = torch.cat([x_dense[nid], feat], dim=1)
            elif self.get_field_nums != "" and self.node_split_idxs[nid] > 0 and x_dense is not None:
                feat = x_dense[nid]
            else:
                feat = f
            feats.append(feat)
        return feats

    @torch.jit.script_method
    def loss(self, last_node_embed, context_x, negatives, targets):
        n = targets.shape[0]
        log_target = torch.log(torch.sigmoid(torch.sum(torch.mul(last_node_embed, context_x[targets]), 1)))
        noise = torch.neg(context_x[negatives])
        noise = self.hidden_dropout(noise)
        sum_log_sampled = torch.sum(torch.log(torch.sigmoid(torch.bmm(noise, last_node_embed.unsqueeze(2)))), 1).squeeze()
        loss = log_target + sum_log_sampled
        return -loss.sum() / n


FLAGS = None


def main():
    gatne = GATNEModel(FLAGS.in_dims, FLAGS.input_embedding_dims, FLAGS.input_field_nums, FLAGS.input_split_idxs,
                       FLAGS.embedding_dim, FLAGS.embedding_u_dim, FLAGS.dim_a, FLAGS.node_types, FLAGS.edge_types,
                       FLAGS.schema, FLAGS.aggregation_type, FLAGS.encode, FLAGS.dropout)
    gatne.save(FLAGS.output_file)


if __name__ == '__main__':
    real_argv = " ".join(sys.argv[1:]).replace("=", " ").split(" ")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dims",
        type=str,
        required=True,
        help="input feature dim for all nodes, such as: 0:10,1:5,2:20")
    parser.add_argument(
        "--input_embedding_dims",
        type=str,
        default="",
        help="embedding dim of all node sparse features, 0:10,1:20,3:20")
    parser.add_argument(
        "--input_field_nums",
        type=str,
        default="",
        help="field num of all node sparse features, such as 0:10,1:20,3:20")
    parser.add_argument(
        "--input_split_idxs",
        type=str,
        default="",
        help="idx split dense and sparse feature for all node, such as 0:10,1:20,3:20")
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=200,
        help="node embedding dimensions. Default is 200.")
    parser.add_argument(
        "--embedding_u_dim",
        type=int,
        default=20,
        help="edge embedding dimensions. Default is 200.")
    parser.add_argument(
        "--dim_a",
        type=int,
        default=20,
        help="attention dimensions. Default is 20.")
    parser.add_argument(
        "--node_types",
        type=str,
        required=True,
        help="all nodes types, such as: 0,1,2")
    parser.add_argument(
        "--edge_types",
        type=str,
        required=True,
        help="all edges types, such as: 0,1,2,3")
    parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="relational schema for Heterogeneous Graphs. Such as 0-0-1,0-1-2")
    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="sum",
        choices=["mean", "sum"],
        help="type of aggregation operation")
    parser.add_argument(
        "--encode",
        type=str,
        default="onehot",
        help="data encode, could be onehot, multihot or dense")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout for negative samples embedding")
    parser.add_argument(
        "--output_file",
        type=str,
        default="gatne.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args(real_argv)
    main()