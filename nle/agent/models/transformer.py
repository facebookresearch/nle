# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn.modules import transformer


class LearnedPositionalEncoder(nn.Module):
    def __init__(self, k, height, width, device):
        super().__init__()

        self.height = height
        self.width = width

        self.enc = torch.randn(height, width, k)

        self.enc = self.enc.div(
            torch.norm(self.enc, p=2, dim=2)[:, :, None].expand_as(self.enc)
        )

        self.mlp = nn.Sequential(
            nn.Linear(2 * k, k), nn.ReLU(), nn.Linear(k, k), nn.ReLU()
        )

        self.enc = nn.Parameter(self.enc, requires_grad=True)[None, :, :, :]

        if device is not None:
            self.enc = self.enc.to(device)

    def forward(self, x):
        x = torch.cat([x, self.enc.expand_as(x)], dim=3)
        x = self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads, height, width, device):
        super().__init__()
        self.N = N
        self.pe = LearnedPositionalEncoder(d_model, height, width, device)
        self.layers = transformer._get_clones(
            transformer.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=d_model
            ),
            N,
        )

    def forward(self, src, mask=None):
        x = src
        x = self.pe(x)

        bs, h, w, k = x.shape

        x = x.view(bs, h * w, k).transpose(1, 0)

        for i in range(self.N):
            x = self.layers[i](x, mask)

        # FIXME: probably slow due to contiguous; we can adapt the rest of the base
        # model to not assume the batch as first dimension
        return x.transpose(1, 0).view(bs, h, w, k).contiguous()
