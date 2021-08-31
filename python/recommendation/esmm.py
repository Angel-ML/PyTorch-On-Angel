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
# !/usr/bin/env python

from __future__ import print_function

import argparse
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List


class ESMM(torch.nn.Module):

    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, fc_dims=[], pos_weight=1,
                 dropout_ratio=0.3, l1_regularization=0.0, l2_regularization=0.0, shared_embedding=0):
        super(ESMM, self).__init__()
        self.loss_fn = torch.nn.BCELoss(weight=torch.tensor([pos_weight]))
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.mats = []
        self.embeddings = []
        self.embeddings_size = []
        self.inputs_size = []
        self.left_mats_len = 0
        self.dropout_ratio = dropout_ratio
        self.L1_regularization = l1_regularization
        self.L2_regularization = l2_regularization
        self.shared_embedding = True if shared_embedding > 0 else False

        # local model do not need real input_dim to init params, so set fake_dim to
        # speed up to produce local pt file.
        fake_input_dim = 10
        if input_dim > 0 and embedding_dim > 0 and n_fields > 0 and fc_dims:
            self.embeddings.append(torch.nn.Parameter(
                torch.zeros(fake_input_dim, embedding_dim)))
            torch.nn.init.xavier_uniform_(self.embeddings[0])
            self.embeddings_size.append(embedding_dim)
            self.inputs_size.append(input_dim)
            if not self.shared_embedding:
                self.embeddings.append(torch.nn.Parameter(
                    torch.zeros(fake_input_dim, embedding_dim)))
                torch.nn.init.xavier_uniform_(self.embeddings[1])
                self.embeddings_size.append(embedding_dim)
                self.inputs_size.append(input_dim)

            dim = n_fields * embedding_dim
            for (index, fc_dim) in enumerate(fc_dims):
                self.mats.append(torch.nn.Parameter(torch.randn(dim, fc_dim)))
                self.mats.append(torch.nn.Parameter(torch.zeros(1, fc_dim)))
                torch.nn.init.kaiming_uniform_(self.mats[index * 2], mode='fan_in', nonlinearity='relu')
                # torch.nn.init.kaiming_uniform_(self.mats[index * 2 + 1], mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(self.mats[index * 2 + 1], 0)
                dim = fc_dim

            self.left_mats_len = len(self.mats)

            dim = n_fields * embedding_dim
            for (index, fc_dim) in enumerate(fc_dims):
                self.mats.append(torch.nn.Parameter(torch.randn(dim, fc_dim)))
                self.mats.append(torch.nn.Parameter(torch.zeros(1, fc_dim)))
                torch.nn.init.kaiming_uniform_(self.mats[index * 2 + self.left_mats_len],
                                               mode='fan_in', nonlinearity='relu')
                # torch.nn.init.kaiming_uniform_(self.mats[index * 2 + 1 + self.left_mats_len],
                #                                mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(self.mats[index * 2 + 1 + self.left_mats_len], 0)
                dim = fc_dim

    def base_forward(self, batch_size, index, embeddings, mats, training):
        # type: (int, Tensor, List[Tensor], List[Tensor], bool) -> Tuple[Tensor, Tensor]
        b = batch_size
        if self.shared_embedding:
            output_left = embeddings[0].view(b, -1)
            output_right = embeddings[0].view(b, -1)
        else:
            output_left = embeddings[0].view(b, -1)
            output_right = embeddings[1].view(b, -1)

        for i in range(int(self.left_mats_len / 2)):
            output_left = output_left.matmul(mats[i * 2]) + mats[i * 2 + 1]
            if i < int(self.left_mats_len / 2) - 1:
                output_left = torch.relu(output_left)
                output_left = F.dropout(output_left, p=self.dropout_ratio, training=training)

        for i in range(int(self.left_mats_len / 2), int(len(mats) / 2)):
            output_right = output_right.matmul(mats[i * 2]) + mats[i * 2 + 1]
            if i < int(len(mats) / 2) - 1:
                output_right = torch.relu(output_right)
                output_right = F.dropout(output_right, p=self.dropout_ratio, training=training)

        return output_left.view(-1), output_right.view(-1)

    def forward_(self, batch_size, training, index, feats, values, targets, embeddings, mats):
        # type: (int, int, Tensor, Tensor, Tensor, Tensor, List[Tensor], List[Tensor]) -> Tensor
        training = True if training else False
        p_cvr, p_ctr = self.base_forward(batch_size, index, embeddings, mats, training)
        p_cvr = torch.sigmoid(p_cvr)
        p_ctr = torch.sigmoid(p_ctr)
        targets = targets.view(batch_size, -1)
        if training:
            return self.loss(p_cvr, p_ctr, targets, mats)  # L1, L2 or non regularization
        else:
            p_ctcvr = torch.mul(p_ctr, p_cvr)
            return torch.cat([p_ctr.view(-1, 1), p_cvr.view(-1, 1), p_ctcvr.view(-1, 1)], dim=1).view(-1)

    def forward(self, batch_size, training, index, feats, values, targets):
        # type: (int, int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        embeds = [F.embedding(feats, self.embeddings[0])]
        if not self.shared_embedding:
            embeds.append(F.embedding(feats, self.embeddings[1]))
        return self.forward_(batch_size, training, index, feats, values, targets,
                             embeds, self.mats)

    def loss(self, p_cvr, p_ctr, targets, mats):
        # type: (Tensor, Tensor, Tensor, List[Tensor]) -> Tensor
        tar_ctr, tar_cvr = targets.chunk(chunks=2, dim=1)
        ctr_loss = self.loss_fn(p_ctr, tar_ctr.view(-1))

        p_ctcvr = torch.mul(p_ctr, p_cvr)
        tar_ctcvr = torch.mul(tar_ctr.view(-1), tar_cvr.view(-1))
        ctcvr_loss = self.loss_fn(p_ctcvr, tar_ctcvr)

        if self.L1_regularization > 0 and self.L2_regularization > 0:
            l1_loss = torch.tensor([0.0])
            l2_loss = torch.tensor([0.0])
            for i in range(int(len(mats) / 2)):
                l1_loss += abs(mats[i * 2]).sum()
                l2_loss += torch.mul(mats[i * 2], mats[i * 2]).sum()
            l1_loss = l1_loss / len(targets)
            l2_loss = l2_loss / len(targets)
            return ctr_loss + ctcvr_loss + l1_loss * self.L1_regularization + l2_loss * self.L2_regularization
        elif self.L1_regularization > 0:
            l1_loss = torch.tensor([0.0])
            for i in range(int(len(mats) / 2)):
                l1_loss += abs(mats[i * 2]).sum()
            l1_loss = l1_loss / len(targets)
            return ctr_loss + ctcvr_loss + l1_loss * self.L1_regularization
        elif self.L2_regularization > 0:
            l2_loss = torch.tensor([0.0])
            for i in range(int(len(mats) / 2)):
                l2_loss += torch.mul(mats[i * 2], mats[i * 2]).sum()
            l2_loss = l2_loss / len(targets)
            return ctr_loss + ctcvr_loss + l2_loss * self.L2_regularization
        else:
            return ctr_loss + ctcvr_loss

    @torch.jit.export
    def get_type(self):
        return "EMBEDDINGS_MATS"

    @torch.jit.export
    def get_name(self):
        return "ESMM"


FLAGS = None


def main():
    esmm = ESMM(FLAGS.input_dim, FLAGS.n_fields, FLAGS.embedding_dim,
                FLAGS.fc_dims, FLAGS.pos_weight, FLAGS.dropout_ratio,
                FLAGS.l1_regularization, FLAGS.l2_regularization, FLAGS.shared_embedding)
    esmm_script_module = torch.jit.script(esmm)
    esmm_script_module.save("esmm.pt")


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
        "--n_fields",
        type=int,
        default=-1,
        help="data num fields."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=-1,
        help="embedding dim."
    )
    parser.add_argument(
        "--fc_dims",
        nargs="+",
        type=int,
        default=-1,
        help="fc layers dim list."
    )
    parser.add_argument("--pos_weight", type=int, default=1, help="pos_weight")
    parser.add_argument("--dropout_ratio", type=float, default=0.0,
                        help="the dropout ratio for bi-interaction layer and the next hidden layers")
    parser.add_argument("--l1_regularization", type=float, default=0.0, help="lambda for L1 regularization")
    parser.add_argument("--l2_regularization", type=float, default=0.0, help="lambda for L2 regularization")

    # parser.add_argument(
    #     "--shared_embedding",
    #     type=bool,
    #     default=False,
    #     help="flag if embeddings are shared."
    # )
    # parser.add_argument(
    #     '--shared_embedding', action='store_true'
    # )
    parser.add_argument(
        "--shared_embedding",
        type=int,
        default=0,
        help="flag if embeddings are shared."
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
