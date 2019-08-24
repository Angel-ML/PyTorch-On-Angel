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
#!/usr/bin/env python
from __future__ import print_function

import torch
import torch.nn.functional as F

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score

import time


import lr, fm, deepfm, deepandwide
import dcn, attention_net
import pnn, attention_fm, xdeepfm



torch.set_num_threads(1)

## load data

input_path = "census_148d_train.libsvm.tmp"
X, Y = load_svmlight_file(input_path, dtype=np.float32)
size, dim = X.shape

## train

# model = lr.LogisticRegression(dim)
# model = fm.FactorizationMachine(dim, 10)
# model = deepfm.DeepFM(dim, 13, 10, [10, 5, 1])
model = deepandwide.DeepAndWide(dim, 13, 10, [10, 5, 1])
# model = attention_net.AttentionNet(dim, 13, 10, [10, 5, 1])
# model = dcn.DCNet(dim, 13, 10, cross_depth=4, deep_layers=[10, 10, 10])
# model = pnn.PNN(dim, 13, 10, [10, 5, 1])
# model = attention_fm.AttentionFM(dim, 13, 10, 10)
# model = xdeepfm.xDeepFM(dim, 13, 10, [10, 5, 5], [128, 128])
model.save("savedmodules/DeepAndWide-model.pt")
model = torch.jit.load("savedmodules/DeepAndWide-model.pt")



optim = torch.optim.Adam(model.parameters(), 0.01)
loss_fn = torch.nn.BCELoss()
batch_size = 30

for epoch in range(10):
    start = 0
    sum_loss = 0.0
    time_start = time.time()
    while start < size:
        optim.zero_grad()
        end = min(start+batch_size, size)
        batch = X[start:end].tocoo()
        y = torch.from_numpy(Y[start:end]).to(torch.float32)

        batch_size, _ = batch.shape
        # batch_size = torch.tensor([batch_size]).to(torch.int32)
        row = torch.from_numpy(batch.row).to(torch.long)
        col = torch.from_numpy(batch.col).to(torch.long)
        data = torch.from_numpy(batch.data)

        y_pred = model(batch_size, row, col, data).view_as(y)

        loss = model.loss(y_pred, y)

        #loss.backward()
        optim.step()

        start += batch_size
        sum_loss += loss.item()* batch_size

    print(sum_loss / size, '%fs' % (time.time() - time_start))


# model.save("model.pt")