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
from torch.nn import functional as F


class ForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions, hidden_dim, input_dim, output_dim):
        super(ForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions

        # TODO: add more layers
        total_input_dim = input_dim + self.num_actions
        self.forward_dynamics = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=-1)
        next_state_emb = self.forward_dynamics(inputs)
        return next_state_emb


class InverseDynamicsNet(nn.Module):
    def __init__(self, num_actions, hidden_dim, input_dim1, input_dim2):
        super(InverseDynamicsNet, self).__init__()
        self.num_actions = num_actions

        # TODO: add more layers
        total_input_dim = input_dim1 + input_dim2  # concat the inputs
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=-1)
        action_logits = self.inverse_dynamics(inputs)
        return action_logits
