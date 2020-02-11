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
from .loop import add_remaining_self_loops, contains_self_loops
from .loop import remove_self_loops, add_self_loops
from .scatter import scatter_add, scatter_mean
from .inits import glorot, zeros, uniform
from .sparse import spmm

__all__ = [
    'scatter_add',
    'scatter_mean',
    'add_remaining_self_loops',
    "contains_self_loops",
    "remove_self_loops",
    "add_self_loops",
    'glorot',
    'zeros',
    'uniform',
    'spmm'
]
