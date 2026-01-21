import torch
import torch.nn.functional as F

import argparse

from nn.conv import GCNConv2  # noqa


class GCN2(torch.jit.ScriptModule):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv2(input_dim, hidden_dim)
        self.conv2 = GCNConv2(hidden_dim, num_classes)

    @torch.jit.script_method
    def forward_(self, x, first_edge_index, second_edge_index):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        x = F.relu(self.conv1(x, second_edge_index))
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, first_edge_index)
        return F.log_softmax(x, dim=1)

    @torch.jit.script_method
    def loss(self, outputs, targets):
        targets = targets.view(-1).to(torch.long)
        return F.nll_loss(outputs, targets)

    @torch.jit.script_method
    def predict_(self, x, first_edge_index, second_edge_index):
        output = self.forward_(x, first_edge_index, second_edge_index)
        return output.max(1)[1]

    def get_training(self):
        return self.training


FLAGS = None


def main():
    gcn = GCN2(FLAGS.input_dim, FLAGS.hidden_dim,
               FLAGS.output_dim)
    gcn.save(FLAGS.output_file)


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
        "--output_file",
        type=str,
        default="graphsage.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
