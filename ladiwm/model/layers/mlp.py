# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn

'''
第一层线性映射 (fc1)

输入维度 in_features

输出维度 hidden_features（通常是 4 * in_features）

起到“扩张通道维度”的作用。

激活函数 (act)

默认 GELU，在 Transformer 里比 ReLU 更常用。

增加非线性表达。

Dropout (drop)

在训练时随机置零部分神经元，起正则化作用。

这里用了两次（激活后、输出后各一次）。

第二层线性映射 (fc2)

输入 hidden_features

输出 out_features（通常和 in_features 相同，保证残差能加回去）。
'''
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
