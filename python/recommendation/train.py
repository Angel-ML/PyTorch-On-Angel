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
# model = deepandwide.DeepAndWide(dim, 13, 10, [10, 5, 1])
# model = attention_net.AttentionNet(dim, 13, 10, [10, 5, 1])
# model = dcn.DCNet(dim, 13, 10, cross_depth=4, deep_layers=[10, 10, 10])
#model = pnn.PNN(dim, 13, 10, [10, 5, 1])
#model = attention_fm.AttentionFM(dim, 13, 10, 10)
#model = xdeepfm.xDeepFM(dim, 13, 10, [10, 5, 5], [128, 128])
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

		y_pred = model(batch_size, row, col, data)

		loss = model.loss(y_pred, y)

		#loss.backward()
		optim.step()

		start += batch_size
		sum_loss += loss.item()* batch_size

	print(sum_loss / size, '%fs' % (time.time() - time_start))


# model.save("model.pt")
