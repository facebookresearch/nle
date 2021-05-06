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

from nle import nethack

from nle.agent.models.embed import GlyphEmbedding
from nle.agent.models.transformer import TransformerEncoder

NUM_GLYPHS = nethack.MAX_GLYPH
NUM_FEATURES = nethack.BLSTATS_SHAPE[0]
PAD_CHAR = 0
NUM_CHARS = 128


class NetHackNet(nn.Module):
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

    def prepare_input(self, inputs):
        # -- [T x B x H x W]
        glyphs = inputs["glyphs"]

        # -- [T x B x F]
        features = inputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        features = features.view(T * B, -1).float()

        return glyphs, features

    def embed_state(self, inputs):
        raise NotImplementedError

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


class RandomNet(NetHackNet):
    def __init__(self, observation_shape, num_actions, flags, device=None):
        super(RandomNet, self).__init__()
        self.num_actions = num_actions
        self.theta = torch.nn.Parameter(torch.zeros(self.num_actions))

    def forward(self, inputs, core_state):
        T, B, *_ = inputs["glyphs"].shape
        zeros = self.theta * 0
        # set logits to 0
        policy_logits = zeros[None, :].expand(T * B, -1)
        # set baseline to 0
        baseline = policy_logits.sum(dim=1).view(-1, B)

        # sample random action
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1).view(
            T, B
        )
        policy_logits = policy_logits.view(T, B, self.num_actions)
        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

    def embed_state(self, inputs):
        raise NotImplementedError


class Crop(nn.Module):
    def __init__(self, height, width, height_target, width_target, device=None):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        self.width_grid = self._step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        self.height_grid = self._step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        if device is not None:
            self.width_grid = self.width_grid.to(device)
            self.height_grid = self.height_grid.to(device)

    def _step_to_range(self, step, num_steps):
        return torch.tensor([step * (i - num_steps // 2) for i in range(num_steps)])

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.

        Args:
           inputs [B x H x W] or [B x H x W x C]
           coordinates [B x 2] x,y coordinates

        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height, "expected %d but found %d" % (
            self.height,
            inputs.shape[1],
        )
        assert inputs.shape[2] == self.width, "expected %d but found %d" % (
            self.width,
            inputs.shape[2],
        )

        permute_results = False
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        else:
            permute_results = True
            inputs = inputs.permute(0, 2, 3, 1)
        inputs = inputs.float()

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

        crop = (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )

        if permute_results:
            # [B x C x H x W] -> [B x H x W x C]
            crop = crop.permute(0, 2, 3, 1)

        return crop


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BaseNet(NetHackNet):
    def __init__(self, observation_shape, num_actions, flags, device):
        super(BaseNet, self).__init__()

        self.flags = flags

        self.observation_shape = observation_shape

        self.H = observation_shape[0]
        self.W = observation_shape[1]

        self.num_actions = num_actions
        self.use_lstm = flags.use_lstm

        self.k_dim = flags.embedding_dim
        self.h_dim = flags.hidden_dim

        self.crop_model = flags.crop_model
        self.crop_dim = flags.crop_dim

        self.num_features = NUM_FEATURES

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim, device)

        self.glyph_type = flags.glyph_type
        self.glyph_embedding = GlyphEmbedding(
            flags.glyph_type, flags.embedding_dim, device, flags.use_index_select
        )

        K = flags.embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        self.Y = 8  # number of output filters
        L = flags.layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [self.Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        if self.crop_model == "transformer":
            self.extract_crop_representation = TransformerEncoder(
                K,
                N=L,
                heads=8,
                height=self.crop_dim,
                width=self.crop_dim,
                device=device,
            )
        elif self.crop_model == "cnn":
            conv_extract_crop = [
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=S,
                    padding=P,
                )
                for i in range(L)
            ]

            self.extract_crop_representation = nn.Sequential(
                *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
            )

        # MESSAGING MODEL
        if "msg" not in flags:
            self.msg_model = "none"
        else:
            self.msg_model = flags.msg.model
            self.msg_hdim = flags.msg.hidden_dim
            self.msg_edim = flags.msg.embedding_dim
        if self.msg_model in ("gru", "lstm", "lt_cnn"):
            # character-based embeddings
            self.char_lt = nn.Embedding(NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR)
        else:
            # forward will set up one-hot inputs for the cnn, no lt needed
            pass

        if self.msg_model.endswith("cnn"):
            # from Zhang et al, 2016
            # Character-level Convolutional Networks for Text Classification
            # https://arxiv.org/abs/1509.01626
            if self.msg_model == "cnn":
                # inputs will be one-hot vectors, as done in paper
                self.conv1 = nn.Conv1d(NUM_CHARS, self.msg_hdim, kernel_size=7)
            elif self.msg_model == "lt_cnn":
                # replace one-hot inputs with learned embeddings
                self.conv1 = nn.Conv1d(self.msg_edim, self.msg_hdim, kernel_size=7)
            else:
                raise NotImplementedError("msg.model == %s", flags.msg.model)

            # remaining convolutions, relus, pools, and a small FC network
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv3
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv4
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv5
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv6
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                nn.ReLU(),
                nn.Linear(2 * self.msg_hdim, self.msg_hdim),
            )  # final output -- [ B x h_dim x 5 ]
        elif self.msg_model in ("gru", "lstm"):

            def rnn(flag):
                return nn.LSTM if flag == "lstm" else nn.GRU

            self.char_rnn = rnn(self.msg_model)(
                self.msg_edim, self.msg_hdim // 2, batch_first=True, bidirectional=True
            )
        elif self.msg_model != "none":
            raise NotImplementedError("msg.model == %s", flags.msg.model)

        self.embed_features = nn.Sequential(
            nn.Linear(self.num_features, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.equalize_input_dim = flags.equalize_input_dim
        if not self.equalize_input_dim:
            # just added up the output dimensions of the input featurizers
            # feature / status dim
            out_dim = self.k_dim
            # CNN over full glyph map
            out_dim += self.H * self.W * self.Y
            if self.crop_model == "transformer":
                out_dim += self.crop_dim ** 2 * K
            elif self.crop_model == "cnn":
                out_dim += self.crop_dim ** 2 * self.Y
            # messaging model
            if self.msg_model != "none":
                out_dim += self.msg_hdim
        else:
            # otherwise, project them all to h_dim
            NUM_INPUTS = 4 if self.msg_model != "none" else 3
            project_hdim = flags.equalize_factor * self.h_dim
            out_dim = project_hdim * NUM_INPUTS

            # set up linear layers for projections
            self.project_feature_dim = nn.Linear(self.k_dim, project_hdim)
            self.project_glyph_dim = nn.Linear(self.H * self.W * self.Y, project_hdim)
            c__2 = self.crop_dim ** 2
            if self.crop_model == "transformer":
                self.project_crop_dim = nn.Linear(c__2 * K, project_hdim)
            elif self.crop_model == "cnn":
                self.project_crop_dim = nn.Linear(c__2 * self.Y, project_hdim)
            if self.msg_model != "none":
                self.project_msg_dim = nn.Linear(self.msg_hdim, project_hdim)

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

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def prepare_input(self, inputs):
        # -- [T x B x H x W]
        T, B, H, W = inputs["glyphs"].shape

        # take our chosen glyphs and merge the time and batch

        glyphs = self.glyph_embedding.prepare_input(inputs)

        # -- [T x B x F]
        features = inputs["blstats"]
        # -- [B' x F]
        features = features.view(T * B, -1).float()

        return glyphs, features

    def forward(self, inputs, core_state, learning=False):
        T, B, *_ = inputs["glyphs"].shape

        glyphs, features = self.prepare_input(inputs)

        # -- [B x 2] x,y coordinates
        coordinates = features[:, :2]

        features = features.view(T * B, -1).float()
        # -- [B x K]
        features_emb = self.embed_features(features)
        if self.equalize_input_dim:
            features_emb = self.project_feature_dim(features_emb)

        assert features_emb.shape[0] == T * B

        reps = [features_emb]

        # -- [B x H' x W']
        crop = self.glyph_embedding.GlyphTuple(
            *[self.crop(g, coordinates) for g in glyphs]
        )
        # -- [B x H' x W' x K]
        crop_emb = self.glyph_embedding(crop)

        if self.crop_model == "transformer":
            # -- [B x W' x H' x K]
            crop_rep = self.extract_crop_representation(crop_emb, mask=None)
        elif self.crop_model == "cnn":
            # -- [B x K x W' x H']
            crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
            # -- [B x W' x H' x K]
            crop_rep = self.extract_crop_representation(crop_emb)
        # -- [B x K']

        crop_rep = crop_rep.view(T * B, -1)
        if self.equalize_input_dim:
            crop_rep = self.project_crop_dim(crop_rep)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self.glyph_embedding(glyphs)
        # glyphs_emb = self.embed(glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)
        if self.equalize_input_dim:
            glyphs_rep = self.project_glyph_dim(glyphs_rep)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        # MESSAGING MODEL
        if self.msg_model != "none":
            # [T x B x 256] -> [T * B x 256]
            messages = inputs["message"].long().view(T * B, -1)
            if self.msg_model == "cnn":
                # convert messages to one-hot, [T * B x 96 x 256]
                one_hot = F.one_hot(messages, num_classes=NUM_CHARS).transpose(1, 2)
                char_rep = self.conv2_6_fc(self.conv1(one_hot.float()))
            elif self.msg_model == "lt_cnn":
                # [ T * B x E x 256 ]
                char_emb = self.char_lt(messages).transpose(1, 2)
                char_rep = self.conv2_6_fc(self.conv1(char_emb))
            else:  # lstm, gru
                char_emb = self.char_lt(messages)
                output = self.char_rnn(char_emb)[0]
                fwd_rep = output[:, -1, : self.h_dim // 2]
                bwd_rep = output[:, 0, self.h_dim // 2 :]
                char_rep = torch.cat([fwd_rep, bwd_rep], dim=1)

            if self.equalize_input_dim:
                char_rep = self.project_msg_dim(char_rep)
            reps.append(char_rep)

        st = torch.cat(reps, dim=1)

        # -- [B x K]
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

        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        output = dict(policy_logits=policy_logits, baseline=baseline, action=action)
        return (output, core_state)
