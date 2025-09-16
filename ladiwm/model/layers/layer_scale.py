# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import torch
from torch import Tensor
from torch import nn

'''
作用

在 残差连接里的子模块输出 上，乘一个 可学习的缩放系数向量 γ (gamma)。

初始值设得很小（通常 1e-5），让网络刚开始训练时几乎等于“只走恒等残差”，避免深层网络一开始就不稳定。

随着训练，γ 会逐渐学到合适的放大倍数，从而让每一层的贡献逐渐增加。
'''

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
