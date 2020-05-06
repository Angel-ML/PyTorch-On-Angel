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
def maybe_dim_size(index, dim_size=None):
    # type: (Tensor, Optional[int]) -> int
    if dim_size is not None:
        return dim_size
    return int(index.max().item()) + 1 if index.numel() > 0 else 0


@torch.jit.script
def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    # type: (Tensor, Tensor, int, Optional[Tensor], Optional[int], int) -> Tuple[Tensor, Tensor, Tensor, int]
    dims = torch.jit.annotate(List[int], [])
    for i in range(src.dim()):
        dims.append(i)
    dim = dims[dim]

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = torch.jit.annotate(List[int], [])
        for i in range(src.dim()):
            index_size.append(1)
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = torch.empty(out_size, dtype=src.dtype, device="cuda")
        out.fill_(fill_value)

    return src, out, index, dim


@torch.jit.script
def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    # type: (Tensor, Tensor, int, Optional[Tensor], Optional[int], int) -> Tensor
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Sums all values from the :attr:`src` tensor into :attr:`out` at the indices
    specified in the :attr:`index` tensor along a given axis :attr:`dim`. For
    each value in :attr:`src`, its output index is specified by its index in
    :attr:`input` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`. If
    multiple indices reference the same location, their **contributions add**.

    Formally, if :attr:`src` and :attr:`index` are n-dimensional tensors with
    size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})` and
    :attr:`dim` = `i`, then :attr:`out` must be an n-dimensional tensor with
    size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`. Moreover, the
    values of :attr:`index` must be between `0` and `out.size(dim) - 1`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j \mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        out (Tensor, optional): The destination tensor. (default: :obj:`None`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor is
            returned. (default: :obj:`None`)
        fill_value (int, optional): If :attr:`out` is not given, automatically
            fill output tensor with :attr:`fill_value`. (default: :obj:`0`)

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_add

        src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out = src.new_zeros((2, 6))

        out = scatter_add(src, index, out=out)

        print(out)

    .. testoutput::

       tensor([[0, 0, 4, 3, 3, 0],
               [2, 4, 4, 0, 0, 0]])
    """
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)


@torch.jit.script
def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    # type: (Tensor, Tensor, int, Optional[Tensor], Optional[int], int) -> Tensor
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/mean.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Averages all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.If multiple indices reference the same location, their
    **contributions average** (`cf.` :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \frac{1}{N_i} \cdot
        \sum_j \mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`. :math:`N_i` indicates the number of indices
    referencing :math:`i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        out (Tensor, optional): The destination tensor. (default: :obj:`None`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor is
            returned. (default: :obj:`None`)
        fill_value (int, optional): If :attr:`out` is not given, automatically
            fill output tensor with :attr:`fill_value`. (default: :obj:`0`)

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_mean

        src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]).float()
        index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out = src.new_zeros((2, 6))

        out = scatter_mean(src, index, out=out)

        print(out)

    .. testoutput::

       tensor([[0.0000, 0.0000, 4.0000, 3.0000, 1.5000, 0.0000],
               [1.0000, 4.0000, 2.0000, 0.0000, 0.0000, 0.0000]])
    """
    out = scatter_add(src, index, dim, out, dim_size, fill_value)
    count = scatter_add(torch.ones_like(src), index, dim, None, out.size(dim))
    return out / count.clamp(min=1)