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

import collections
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


from nle import nethack

from util.id_pairs import id_pairs_table
import numpy as np

NUM_GLYPHS = nethack.MAX_GLYPH
NUM_FEATURES = nethack.BLSTATS_SHAPE[0]
PAD_CHAR = 0
NUM_CHARS = 256


def get_action_space_mask(action_space, reduced_action_space):
    mask = np.array([int(a in reduced_action_space) for a in action_space])
    return torch.Tensor(mask)


def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)


def select(embedding_layer, x, use_index_select):
    """Use index select instead of default forward to possible speed up embedding."""
    if use_index_select:
        out = embedding_layer.weight.index_select(0, x.view(-1))
        # handle reshaping x to 1-d and output back to N-d
        return out.view(x.shape + (-1,))
    else:
        return embedding_layer(x)


class NetHackNet(nn.Module):
    """This base class simply provides a skeleton for running with torchbeast."""

    AgentOutput = collections.namedtuple("AgentOutput", "action policy_logits baseline")

    def __init__(self):
        super(NetHackNet, self).__init__()
        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))

    def forward(self, inputs, core_state):
        raise NotImplementedError

    def initial_state(self, batch_size=1):
        return ()

    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        return torch.sqrt(self.reward_m2 / self.reward_count)


class BaselineNet(NetHackNet):
    """This model combines the encodings of the glyphs, top line message and
    blstats into a single fixed-size representation, which is then passed to
    an LSTM core before generating a policy and value head for use in an IMPALA
    like architecture.

    This model was based on 'neurips2020release' tag on the NLE repo, itself
    based on Kuttler et al, 2020
    The NetHack Learning Environment
    https://arxiv.org/abs/2006.13760
    """

    def __init__(self, observation_shape, action_space, flags, device):
        super(BaselineNet, self).__init__()

        self.flags = flags

        self.observation_shape = observation_shape
        self.num_actions = len(action_space)

        self.H = observation_shape[0]
        self.W = observation_shape[1]

        self.use_lstm = flags.use_lstm
        self.h_dim = flags.hidden_dim

        # GLYPH + CROP MODEL
        self.glyph_model = GlyphEncoder(flags, self.H, self.W, flags.crop_dim, device)

        # MESSAGING MODEL
        self.msg_model = MessageEncoder(
            flags.msg.hidden_dim, flags.msg.embedding_dim, device
        )

        # BLSTATS MODEL
        self.blstats_model = BLStatsEncoder(NUM_FEATURES, flags.embedding_dim)

        out_dim = (
            self.blstats_model.hidden_dim
            + self.glyph_model.hidden_dim
            + self.msg_model.hidden_dim
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)

        self.policy = nn.Linear(self.h_dim, self.num_actions)
        self.baseline = nn.Linear(self.h_dim, 1)

        if flags.restrict_action_space:
            reduced_space = nethack.USEFUL_ACTIONS
            logits_mask = get_action_space_mask(action_space, reduced_space)
            self.policy_logits_mask = nn.parameter.Parameter(
                logits_mask, requires_grad=False
            )

    def initial_state(self, batch_size=1):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state, learning=False):
        T, B, H, W = inputs["glyphs"].shape

        reps = []

        # -- [B' x K] ; B' == (T x B)
        glyphs_rep = self.glyph_model(inputs)
        reps.append(glyphs_rep)

        # -- [B' x K]
        char_rep = self.msg_model(inputs)
        reps.append(char_rep)

        # -- [B' x K]
        features_emb = self.blstats_model(inputs)
        reps.append(features_emb)

        # -- [B' x K]
        st = torch.cat(reps, dim=1)

        # -- [B' x K]
        st = self.fc(st)

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * t for t in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        # -- [B' x A]
        policy_logits = self.policy(core_output)

        # -- [B' x 1]
        baseline = self.baseline(core_output)

        if self.flags.restrict_action_space:
            policy_logits = policy_logits * self.policy_logits_mask + (
                (1 - self.policy_logits_mask) * -1e10
            )

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        output = dict(policy_logits=policy_logits, baseline=baseline, action=action)
        return (output, core_state)


class GlyphEncoder(nn.Module):
    """This glyph encoder first breaks the glyphs (integers up to 6000) to a
    more structured representation based on the qualities of the glyph: chars,
    colors, specials, groups and subgroup ids..
       Eg: invisible hell-hound: char (d), color (red), specials (invisible),
                                 group (monster) subgroup id (type of monster)
       Eg: lit dungeon floor: char (.), color (white), specials (none),
                              group (dungeon) subgroup id (type of dungeon)

    An embedding is provided for each of these, and the embeddings are
    concatenated, before encoding with a number of CNN layers.  This operation
    is repeated with a crop of the structured reprentations taken around the
    characters position, and the two representations are concatenated
    before returning.
    """

    def __init__(self, flags, rows, cols, crop_dim, device=None):
        super(GlyphEncoder, self).__init__()

        self.crop = Crop(rows, cols, crop_dim, crop_dim, device)
        K = flags.embedding_dim  # number of input filters
        L = flags.layers  # number of convnet layers

        assert (
            K % 8 == 0
        ), "This glyph embedding format needs embedding dim to be multiple of 8"
        unit = K // 8
        self.chars_embedding = nn.Embedding(256, 2 * unit)
        self.colors_embedding = nn.Embedding(16, unit)
        self.specials_embedding = nn.Embedding(256, unit)

        self.id_pairs_table = nn.parameter.Parameter(
            torch.from_numpy(id_pairs_table()), requires_grad=False
        )
        num_groups = self.id_pairs_table.select(1, 1).max().item() + 1
        num_ids = self.id_pairs_table.select(1, 0).max().item() + 1

        self.groups_embedding = nn.Embedding(num_groups, unit)
        self.ids_embedding = nn.Embedding(num_ids, 3 * unit)

        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        self.output_filters = 8

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [self.output_filters]

        h, w, c = rows, cols, crop_dim
        conv_extract, conv_extract_crop = [], []
        for i in range(L):
            conv_extract.append(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=S,
                    padding=P,
                )
            )
            conv_extract.append(nn.ELU())

            conv_extract_crop.append(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=S,
                    padding=P,
                )
            )
            conv_extract_crop.append(nn.ELU())

            # Keep track of output shapes
            h = conv_outdim(h, F, P, S)
            w = conv_outdim(w, F, P, S)
            c = conv_outdim(c, F, P, S)

        self.hidden_dim = (h * w + c * c) * self.output_filters
        self.extract_representation = nn.Sequential(*conv_extract)
        self.extract_crop_representation = nn.Sequential(*conv_extract_crop)
        self.select = lambda emb, x: select(emb, x, flags.use_index_select)

    def glyphs_to_ids_groups(self, glyphs):
        T, B, H, W = glyphs.shape
        ids_groups = self.id_pairs_table.index_select(0, glyphs.view(-1).long())
        ids = ids_groups.select(1, 0).view(T, B, H, W).long()
        groups = ids_groups.select(1, 1).view(T, B, H, W).long()
        return [ids, groups]

    def forward(self, inputs):
        T, B, H, W = inputs["glyphs"].shape
        ids, groups = self.glyphs_to_ids_groups(inputs["glyphs"])

        glyph_tensors = [
            self.select(self.chars_embedding, inputs["chars"].long()),
            self.select(self.colors_embedding, inputs["colors"].long()),
            self.select(self.specials_embedding, inputs["specials"].long()),
            self.select(self.groups_embedding, groups),
            self.select(self.ids_embedding, ids),
        ]

        glyphs_emb = torch.cat(glyph_tensors, dim=-1)
        glyphs_emb = rearrange(glyphs_emb, "T B H W K -> (T B) K H W")

        coordinates = inputs["blstats"].view(T * B, -1).float()[:, :2]
        crop_emb = self.crop(glyphs_emb, coordinates)

        glyphs_rep = self.extract_representation(glyphs_emb)
        glyphs_rep = rearrange(glyphs_rep, "B C H W -> B (C H W)")
        assert glyphs_rep.shape[0] == T * B

        crop_rep = self.extract_crop_representation(crop_emb)
        crop_rep = rearrange(crop_rep, "B C H W -> B (C H W)")
        assert crop_rep.shape[0] == T * B

        st = torch.cat([glyphs_rep, crop_rep], dim=1)
        return st


class MessageEncoder(nn.Module):
    """This model encodes the the topline message into a fixed size representation.

    It works by using a learnt embedding for each character before passing the
    embeddings through 6 CNN layers.

    Inspired by Zhang et al, 2016
    Character-level Convolutional Networks for Text Classification
    https://arxiv.org/abs/1509.01626
    """

    def __init__(self, hidden_dim, embedding_dim, device=None):
        super(MessageEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.msg_edim = embedding_dim

        self.char_lt = nn.Embedding(NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR)
        self.conv1 = nn.Conv1d(self.msg_edim, self.hidden_dim, kernel_size=7)
        self.conv2_6_fc = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # conv2
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # conv3
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3),
            nn.ReLU(),
            # conv4
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3),
            nn.ReLU(),
            # conv5
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3),
            nn.ReLU(),
            # conv6
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # fc receives -- [ B x h_dim x 5 ]
            Flatten(),
            nn.Linear(5 * self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )  # final output -- [ B x h_dim x 5 ]

    def forward(self, inputs):
        T, B, *_ = inputs["message"].shape
        messages = inputs["message"].long().view(T * B, -1)
        # [ T * B x E x 256 ]
        char_emb = self.char_lt(messages).transpose(1, 2)
        char_rep = self.conv2_6_fc(self.conv1(char_emb))
        return char_rep


class BLStatsEncoder(nn.Module):
    """This model encodes the bottom line stats into a fixed size representation.

    It works by simply using two fully-connected layers with ReLU activations.
    """

    def __init__(self, num_features, hidden_dim):
        super(BLStatsEncoder, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embed_features = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, inputs):
        T, B, *_ = inputs["blstats"].shape

        features = inputs["blstats"]
        # -- [B' x F]
        features = features.view(T * B, -1).float()
        # -- [B x K]
        features_emb = self.embed_features(features)

        assert features_emb.shape[0] == T * B
        return features_emb


class Crop(nn.Module):
    def __init__(self, height, width, height_target, width_target, device=None):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target

        width_grid = self._step_to_range(2 / (self.width - 1), self.width_target)
        self.width_grid = width_grid[None, :].expand(self.height_target, -1)

        height_grid = self._step_to_range(2 / (self.height - 1), height_target)
        self.height_grid = height_grid[:, None].expand(-1, self.width_target)

        if device is not None:
            self.width_grid = self.width_grid.to(device)
            self.height_grid = self.height_grid.to(device)

    def _step_to_range(self, step, num_steps):
        return torch.tensor([step * (i - num_steps // 2) for i in range(num_steps)])

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.

        Args:
           inputs [B x H x W] or [B x C x H x W]
           coordinates [B x 2] x,y coordinates

        Returns:
           [B x C x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1).float()

        assert inputs.shape[2] == self.height, "expected %d but found %d" % (
            self.height,
            inputs.shape[2],
        )
        assert inputs.shape[3] == self.width, "expected %d but found %d" % (
            self.width,
            inputs.shape[3],
        )

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        crop = torch.round(F.grid_sample(inputs, grid, align_corners=True)).squeeze(1)
        return crop


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
