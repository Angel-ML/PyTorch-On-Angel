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

from .sage_conv import SAGEConv
from .gcn_conv import GCNConv, GCNConv2
from .sage_conv import SAGEConv, SAGEConv2, SAGEConv3, SAGEConv4, EdgeSAGEConv
from .bisage_conv import BiSAGEConv
from .igmc_conv import IGMCConv
from .hgat_conv import HGATConv
from .rgcn_conv import RGCNConv
from .han_conv import HANConv

__all__ = [
    'GCNConv',
    'GCNConv2',
    'SAGEConv',
    'SAGEConv2',
    'SAGEConv3',
    'SAGEConv4',
    'EdgeSAGEConv',
    'RGCNConv',
    'IGMCConv',
    'BiSAGEConv',
    'HGATConv',
    'HANConv',
]
