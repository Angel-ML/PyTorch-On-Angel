#!/usr/bin/env python

import torch
import torch.nn.functional as F
from nn.conv import GCNConv

class Net(torch.jit.ScriptModule):
    def __init__(self, input_dim, hidden, output_dim):
        super(Net, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)






from utils import scatter_mean

src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]).float()
index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
out = src.new_zeros((2, 6))
out = scatter_mean(src, index, out=out)
#print(out)


# class LinearFunction(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input, weight):
#         ctx.save_for_backward(input, weight)
#         output = input.mm(weight)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight = ctx.saved_tensors
#         grad_input = grad_weight = None

#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.mm(weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.mm(input)

#         return grad_input, grad_weight


# def linear(a, b):
#     return LinearFunction.apply(a, b)

# trace_linear = torch.jit.trace(linear, (torch.randn((2, 2)), torch.randn((2, 2))))

# @torch.jit.script
# def bar(x, y):
#     return trace_linear(x, y)


# a = torch.randn((2, 2), requires_grad=True)
# b = torch.randn((2, 2), requires_grad=True)

# c = bar(a, b).sum()
# c.backward()


from torch_scatter import scatter_max

src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]], dtype=torch.long)
index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
out = src.new_zeros((2, 6), dtype=torch.long)

# out, argmax = scatter_max(src, index, out=out)
# print(out)
# print(argmax)

from torch.autograd import Function

from torch_scatter.utils.ext import get_func
from torch_scatter.utils.gen import gen

class ScatterMax(Function):
    @staticmethod
    def forward(ctx, out, src, index, dim):
        arg = torch.zeros(out.size(), dtype=torch.long)
        arg.fill_(-1)
        # arg = index.new_full(out.size(), -1)
        func = get_func('scatter_max', src)
        func(src, index, out, arg, dim)

        # ctx.mark_dirty(out)
        ctx.dim = dim
        ctx.save_for_backward(index, arg)

        return out, arg

    @staticmethod
    def backward(ctx, grad_out, grad_arg):
        index, arg = ctx.saved_tensors

        grad_src = None
        if ctx.needs_input_grad[1]:
            grad_src = grad_out.new_zeros(index.size())
            func = get_func('index_backward', grad_out)
            func(grad_out, index, arg, grad_src, ctx.dim)

        return None, grad_src, None, None



def scatter_max_wrap(src, index, out):
    src, out, index, dim = gen(src, index, -1, out, None, 0)
    return ScatterMax.apply(out, src, index, dim)


def for_func(a, b, c):
    c = int(c.item())
    for i in range(c):
        a.add_(b)
    return a

for_func_trace = torch.jit.trace(for_func, (torch.ones(1), torch.ones(1), torch.tensor([10])))

# scatter_max_trace = torch.jit.trace(scatter_max_wrap, (src, index, out))

# @torch.jit.script
# def scatter_max(out, src, index):
#     # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
#     return scatter_max_trace(src, index, out)

# out, argmax = scatter_max(out, src, index)
# print(out)
# print(argmax)

def inplace_foo(a, b):
    a.add_(b)
    return a

inplace_foo_trace = torch.jit.trace(inplace_foo, (torch.ones(10), torch.randn(10)))


@torch.jit.script
def mask(a):
    mask = a == 0
    print(mask)
    # a[mask] = 0
    a.masked_fill_(mask, 2)
    return a

a = torch.ones(5)
a[1] = 0
a = mask(a)
print(a)









