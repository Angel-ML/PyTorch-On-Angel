import torch
from torch import nn
import torch.nn.functional as F

import argparse


class GAMLP(torch.jit.ScriptModule):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout, hops):
        super(GAMLP, self).__init__()

        self.W0 = nn.Linear((hops + 1) * input_dim, hidden_dim)
        self.W1 = nn.Linear(input_dim, 1)
        self.W2 = nn.Linear(hidden_dim, 1)

        self.W3 = nn.Linear(input_dim, hidden_dim)
        self.W4 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(p=dropout)

        self.hops = hops
        self.input_dim = input_dim

    @torch.jit.script_method
    # feature_list is a 2-d tensor of shape (*, (hops + 1) * input_dim)
    def forward_(self, x, first_edge_index):
        # type: (Tensor, Tensor) -> Tensor
        node_list = torch._unique(first_edge_index[0], sorted=True)[0]
        feature_list = x[node_list]

        r = F.relu(self.W0(feature_list))
        W1M = self.W1(feature_list.view(
            feature_list.shape[0], self.hops + 1, self.input_dim)).view(
            feature_list.shape[0], self.hops + 1)
        W2r = self.W2(r)
        phi = torch.sigmoid(W1M + W2r)
        W = F.softmax(phi, dim=1).view(phi.shape[0], 1, -1)
        c_msg = torch.matmul(W, feature_list.view(
            feature_list.shape[0], self.hops + 1, self.input_dim))
        c_msg = c_msg.view(c_msg.shape[0], -1)

        hid = F.relu(self.W3(c_msg))
        hid = self.dropout(hid)
        output = self.W4(hid)
        return output

    @torch.jit.script_method
    def loss(self, outputs, targets):
        # type: (Tensor, Tensor) -> Tensor
        targets = targets.view(-1).to(torch.long)
        return F.cross_entropy(outputs, targets)

    @torch.jit.script_method
    def predict_(self, x, first_edge_index):
        # type: (Tensor, Tensor) -> Tensor
        output = self.forward_(x, first_edge_index)
        return output.max(1)[1]

    @torch.jit.script_method
    def embedding_(self, x, first_edge_index):
        # type: (Tensor, Tensor) -> Tensor
        node_list = torch._unique(first_edge_index[0], sorted=True)[0]
        feature_list = x[node_list]

        r = F.relu(self.W0(feature_list))
        W1M = self.W1(feature_list.view(
            feature_list.shape[0], self.hops + 1, self.input_dim)).view(
            feature_list.shape[0], self.hops + 1)
        W2r = self.W2(r)
        phi = torch.sigmoid(W1M + W2r)
        W = F.softmax(phi, dim=1).view(phi.shape[0], 1, -1)
        c_msg = torch.matmul(W, feature_list.view(
            feature_list.shape[0], self.hops + 1, self.input_dim))
        c_msg = c_msg.view(c_msg.shape[0], -1)

        hid = F.relu(self.W3(c_msg))
        return hid

    @torch.jit.script_method
    def embedding_predict_(self, embedding):
        # type: (Tensor) -> Tensor
        return self.W4(self.dropout(embedding))

    def get_training(self):
        return self.training


FLAGS = None


def main():
    gamlp = GAMLP(FLAGS.input_dim, FLAGS.hidden_dim,
                FLAGS.output_dim, FLAGS.dropout,
                FLAGS.hops)
    gamlp.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="input dimention of node features")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=-1,
        help="hidden dimension of graphsage convolution layer")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=-1,
        help="output dimension, the number of labels")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="the percentage of dropout")
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="the propagation depth")
    parser.add_argument(
        "--output_file",
        type=str,
        default="gamlp.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
