#!/usr/bin/env python

from __future__ import print_function

import argparse
import torch

from utils import scatter_mean


class Aggregator(torch.jit.ScriptModule):

    def __init__(self):
        super(Aggregator, self).__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)

    @torch.jit.script_method
    def embedding_(self, x, edge_index):
        # self loops are contained in edge_index
        row, col = edge_index[0], edge_index[1]
        smoothed_features = scatter_mean(
            x[col], row, dim=0)  # do not set dim_size
        return smoothed_features


FLAGS = None


def main():
    sage = Aggregator()
    sage.save(FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="aggregator.pt",
        help="output file name")
    FLAGS, unparsed = parser.parse_known_args()
    main()
