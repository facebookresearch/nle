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

import copy
import logging
import math
import torch
from torch import nn
from torch.nn import functional as F

from nle.agent.models.base import BaseNet, PAD_CHAR, NUM_CHARS
from nle.agent.models.embed import GlyphEmbedding
from nle.agent.models.dynamics import ForwardDynamicsNet, InverseDynamicsNet


class IntrinsicRewardNet(BaseNet):
    def __init__(self, observation_shape, num_actions, flags, device):
        super(IntrinsicRewardNet, self).__init__(
            observation_shape, num_actions, flags, device
        )
        self.register_buffer("intrinsic_sum", torch.zeros(()))
        self.register_buffer("intrinsic_m2", torch.zeros(()))
        self.register_buffer("intrinsic_count", torch.zeros(()).fill_(1e-8))

        self.intrinsic_input = flags.int.input

        self.int_baseline = nn.Linear(self.h_dim, 1)

    def intrinsic_enabled(self):
        return True

    @torch.no_grad()
    def update_intrinsic_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.intrinsic_sum / self.intrinsic_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
            (self.intrinsic_count * new_count)
            / (self.intrinsic_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.intrinsic_count += new_count
        self.intrinsic_sum += new_sum
        self.intrinsic_m2 += new_m2

    @torch.no_grad()
    def get_intrinsic_std(self):
        """Returns standard deviation of the running mean of the intrinsic reward."""
        return torch.sqrt(self.intrinsic_m2 / self.intrinsic_count)


class RNDNet(IntrinsicRewardNet):
    def __init__(self, observation_shape, num_actions, flags, device):
        super(RNDNet, self).__init__(observation_shape, num_actions, flags, device)

        if self.equalize_input_dim:
            raise NotImplementedError("rnd model does not support equalize_input_dim")

        Y = 8  # number of output filters

        # IMPLEMENTED HERE: RND net using the default feature extractor
        self.rndtgt_embed = GlyphEmbedding(
            flags.glyph_type, flags.embedding_dim, device, flags.use_index_select
        ).requires_grad_(False)
        self.rndprd_embed = GlyphEmbedding(
            flags.glyph_type, flags.embedding_dim, device, flags.use_index_select
        )

        if self.intrinsic_input not in ("crop_only", "glyph_only", "full"):
            raise NotImplementedError("RND input type %s" % self.intrinsic_input)

        rnd_out_dim = 0
        if self.intrinsic_input in ("crop_only", "full"):
            self.rndtgt_extract_crop_representation = copy.deepcopy(
                self.extract_crop_representation
            ).requires_grad_(False)
            self.rndprd_extract_crop_representation = copy.deepcopy(
                self.extract_crop_representation
            )

            rnd_out_dim += self.crop_dim ** 2 * Y  # crop dim

        if self.intrinsic_input in ("full", "glyph_only"):
            self.rndtgt_extract_representation = copy.deepcopy(
                self.extract_representation
            ).requires_grad_(False)
            self.rndprd_extract_representation = copy.deepcopy(
                self.extract_representation
            )
            rnd_out_dim += self.H * self.W * Y  # glyph dim

            if self.intrinsic_input == "full":
                self.rndtgt_embed_features = nn.Sequential(
                    nn.Linear(self.num_features, self.k_dim),
                    nn.ELU(),
                    nn.Linear(self.k_dim, self.k_dim),
                    nn.ELU(),
                ).requires_grad_(False)
                self.rndprd_embed_features = nn.Sequential(
                    nn.Linear(self.num_features, self.k_dim),
                    nn.ELU(),
                    nn.Linear(self.k_dim, self.k_dim),
                    nn.ELU(),
                )
                rnd_out_dim += self.k_dim  # feature dim

        if self.intrinsic_input == "full" and self.msg_model != "none":
            # we only implement the lt_cnn msg model for RND for simplicity & speed
            if self.msg_model != "lt_cnn":
                logging.warning(
                    "msg.model set to %s, but RND overriding to lt_cnn for its input--"
                    "so the policy and RND are using different models for the messages"
                    % self.msg_model
                )

            self.rndtgt_char_lt = nn.Embedding(
                NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR
            ).requires_grad_(False)
            self.rndprd_char_lt = nn.Embedding(
                NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR
            )

            # similar to Zhang et al, 2016
            # Character-level Convolutional Networks for Text Classification
            # https://arxiv.org/abs/1509.01626
            # replace one-hot inputs with learned embeddings
            self.rndtgt_conv1 = nn.Conv1d(
                self.msg_edim, self.msg_hdim, kernel_size=7
            ).requires_grad_(False)
            self.rndprd_conv1 = nn.Conv1d(self.msg_edim, self.msg_hdim, kernel_size=7)

            # remaining convolutions, relus, pools, and a small FC network
            self.rndtgt_conv2_6_fc = copy.deepcopy(self.conv2_6_fc).requires_grad_(
                False
            )
            self.rndprd_conv2_6_fc = copy.deepcopy(self.conv2_6_fc)
            rnd_out_dim += self.msg_hdim

        self.rndtgt_fc = nn.Sequential(  # matching RND paper making this smaller
            nn.Linear(rnd_out_dim, self.h_dim)
        ).requires_grad_(False)
        self.rndprd_fc = nn.Sequential(  # matching RND paper making this bigger
            nn.Linear(rnd_out_dim, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        modules_to_init = [
            self.rndtgt_embed,
            self.rndprd_embed,
            self.rndtgt_fc,
            self.rndprd_fc,
        ]

        SQRT_2 = math.sqrt(2)

        def init(p):
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                # init method used in paper
                nn.init.orthogonal_(p.weight, SQRT_2)
                p.bias.data.zero_()
            if isinstance(p, nn.Embedding):
                nn.init.orthogonal_(p.weight, SQRT_2)

        # manually init all to orthogonal dist

        if self.intrinsic_input in ("full", "crop_only"):
            modules_to_init.append(self.rndtgt_extract_crop_representation)
            modules_to_init.append(self.rndprd_extract_crop_representation)
        if self.intrinsic_input in ("full", "glyph_only"):
            modules_to_init.append(self.rndtgt_extract_representation)
            modules_to_init.append(self.rndprd_extract_representation)
        if self.intrinsic_input == "full":
            modules_to_init.append(self.rndtgt_embed_features)
            modules_to_init.append(self.rndprd_embed_features)
            if self.msg_model != "none":
                modules_to_init.append(self.rndtgt_conv2_6_fc)
                modules_to_init.append(self.rndprd_conv2_6_fc)

        for m in modules_to_init:
            for p in m.modules():
                init(p)

    def forward(self, inputs, core_state, learning=False):
        if not learning:
            # no need to calculate RND outputs when not in learn step
            return super(RNDNet, self).forward(inputs, core_state, learning)
        T, B, *_ = inputs["glyphs"].shape

        glyphs, features = self.prepare_input(inputs)

        # -- [B x 2] x,y coordinates
        coordinates = features[:, :2]

        features = features.view(T * B, -1).float()
        # -- [B x K]
        features_emb = self.embed_features(features)

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
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self.glyph_embedding(glyphs)
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

        # TARGET NETWORK
        with torch.no_grad():
            if self.intrinsic_input == "crop_only":
                tgt_crop_emb = self.rndtgt_embed(crop).transpose(1, 3)
                tgt_crop_rep = self.rndtgt_extract_crop_representation(tgt_crop_emb)
                tgt_st = self.rndtgt_fc(tgt_crop_rep.view(T * B, -1))
            elif self.intrinsic_input == "glyph_only":
                tgt_glyphs_emb = self.rndtgt_embed(glyphs).transpose(1, 3)
                tgt_glyphs_rep = self.rndtgt_extract_representation(tgt_glyphs_emb)
                tgt_st = self.rndtgt_fc(tgt_glyphs_rep.view(T * B, -1))
            else:  # full
                tgt_reps = []
                tgt_feats = self.rndtgt_embed_features(features)
                tgt_reps.append(tgt_feats)

                tgt_crop_emb = self.rndtgt_embed(crop).transpose(1, 3)
                tgt_crop_rep = self.rndtgt_extract_crop_representation(tgt_crop_emb)
                tgt_reps.append(tgt_crop_rep.view(T * B, -1))

                tgt_glyphs_emb = self.rndtgt_embed(glyphs).transpose(1, 3)
                tgt_glyphs_rep = self.rndtgt_extract_representation(tgt_glyphs_emb)
                tgt_reps.append(tgt_glyphs_rep.view(T * B, -1))

                if self.msg_model != "none":
                    tgt_char_emb = self.rndtgt_char_lt(messages).transpose(1, 2)
                    tgt_char_rep = self.rndtgt_conv2_6_fc(
                        self.rndprd_conv1(tgt_char_emb)
                    )
                    tgt_reps.append(tgt_char_rep)

                tgt_st = self.rndtgt_fc(torch.cat(tgt_reps, dim=1))

        # PREDICTOR NETWORK
        if self.intrinsic_input == "crop_only":
            prd_crop_emb = self.rndprd_embed(crop).transpose(1, 3)
            prd_crop_rep = self.rndprd_extract_crop_representation(prd_crop_emb)
            prd_st = self.rndprd_fc(prd_crop_rep.view(T * B, -1))
        elif self.intrinsic_input == "glyph_only":
            prd_glyphs_emb = self.rndprd_embed(glyphs).transpose(1, 3)
            prd_glyphs_rep = self.rndprd_extract_representation(prd_glyphs_emb)
            prd_st = self.rndprd_fc(prd_glyphs_rep.view(T * B, -1))
        else:  # full
            prd_reps = []
            prd_feats = self.rndprd_embed_features(features)
            prd_reps.append(prd_feats)

            prd_crop_emb = self.rndprd_embed(crop).transpose(1, 3)
            prd_crop_rep = self.rndprd_extract_crop_representation(prd_crop_emb)
            prd_reps.append(prd_crop_rep.view(T * B, -1))

            prd_glyphs_emb = self.rndprd_embed(glyphs).transpose(1, 3)
            prd_glyphs_rep = self.rndprd_extract_representation(prd_glyphs_emb)
            prd_reps.append(prd_glyphs_rep.view(T * B, -1))

            if self.msg_model != "none":
                prd_char_emb = self.rndprd_char_lt(messages).transpose(1, 2)
                prd_char_rep = self.rndprd_conv2_6_fc(self.rndprd_conv1(prd_char_emb))
                prd_reps.append(prd_char_rep)

            prd_st = self.rndprd_fc(torch.cat(prd_reps, dim=1))

        assert tgt_st.size() == prd_st.size()

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

        output = dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
            target=tgt_st.view(T, B, -1),
            predicted=prd_st.view(T, B, -1),
            int_baseline=self.int_baseline(core_output).view(T, B),
        )
        return (output, core_state)


class RIDENet(IntrinsicRewardNet):
    def __init__(self, observation_shape, num_actions, flags, device):
        super(RIDENet, self).__init__(observation_shape, num_actions, flags, device)

        if flags.msg.model != "none":
            raise NotImplementedError(
                "model=%s + msg.model=%s" % (flags.model, flags.msg.model)
            )

        self.forward_dynamics_model = ForwardDynamicsNet(
            num_actions, flags.ride.hidden_dim, flags.hidden_dim, flags.hidden_dim
        )
        self.inverse_dynamics_model = InverseDynamicsNet(
            num_actions, flags.ride.hidden_dim, flags.hidden_dim, flags.hidden_dim
        )

        Y = 8  # number of output filters

        # IMPLEMENTED HERE: RIDE net using the default feature extractor
        self.ride_embed = GlyphEmbedding(
            flags.glyph_type, flags.embedding_dim, device, flags.use_index_select
        )

        if self.intrinsic_input not in ("crop_only", "glyph_only", "full"):
            raise NotImplementedError("RIDE input type %s" % self.intrinsic_input)

        ride_out_dim = 0
        if self.intrinsic_input in ("crop_only", "full"):
            self.ride_extract_crop_representation = copy.deepcopy(
                self.extract_crop_representation
            )
            ride_out_dim += self.crop_dim ** 2 * Y  # crop dim

        if self.intrinsic_input in ("full", "glyph_only"):
            self.ride_extract_representation = copy.deepcopy(
                self.extract_representation
            )
            ride_out_dim += self.H * self.W * Y  # glyph dim

            if self.intrinsic_input == "full":
                self.ride_embed_features = nn.Sequential(
                    nn.Linear(self.num_features, self.k_dim),
                    nn.ELU(),
                    nn.Linear(self.k_dim, self.k_dim),
                    nn.ELU(),
                )
                ride_out_dim += self.k_dim  # feature dim

        self.ride_fc = nn.Sequential(
            nn.Linear(ride_out_dim, self.h_dim),
            # nn.ELU(),
            # nn.Linear(self.h_dim, self.h_dim),
            # nn.ELU(),
            # nn.Linear(self.h_dim, self.h_dim),
        )

        # reinitialize all deep-copied layers
        modules_to_init = []
        if self.intrinsic_input in ("full", "crop_only"):
            modules_to_init.append(self.ride_extract_crop_representation)
        if self.intrinsic_input in ("full", "glyph_only"):
            modules_to_init.append(self.ride_extract_representation)

        for m in modules_to_init:
            for p in m.modules():
                if isinstance(p, nn.Conv2d):
                    p.reset_parameters()

    def forward(self, inputs, core_state, learning=False):
        if not learning:
            # no need to calculate RIDE outputs when not in learn step
            return super(RIDENet, self).forward(inputs, core_state, learning)

        T, B, *_ = inputs["glyphs"].shape

        glyphs, features = self.prepare_input(inputs)

        # -- [B x 2] x,y coordinates
        coordinates = features[:, :2]

        features = features.view(T * B, -1).float()
        # -- [B x K]
        features_emb = self.embed_features(features)

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
        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)

        # PREDICTOR NETWORK
        if self.intrinsic_input == "crop_only":
            ride_crop_emb = self.ride_embed(crop).transpose(1, 3)
            ride_crop_rep = self.ride_extract_crop_representation(ride_crop_emb)
            ride_st = self.ride_fc(ride_crop_rep.view(T * B, -1))
        elif self.intrinsic_input == "glyph_only":
            ride_glyphs_emb = self.ride_embed(glyphs).transpose(1, 3)
            ride_glyphs_rep = self.ride_extract_representation(ride_glyphs_emb)
            ride_st = self.ride_fc(ride_glyphs_rep.view(T * B, -1))
        else:  # full
            ride_reps = []
            ride_feats = self.ride_embed_features(features)
            ride_reps.append(ride_feats)

            ride_crop_emb = self.ride_embed(crop).transpose(1, 3)
            ride_crop_rep = self.ride_extract_crop_representation(ride_crop_emb)
            ride_reps.append(ride_crop_rep.view(T * B, -1))

            ride_glyphs_emb = self.ride_embed(glyphs).transpose(1, 3)
            ride_glyphs_rep = self.ride_extract_representation(ride_glyphs_emb)
            ride_reps.append(ride_glyphs_rep.view(T * B, -1))

            ride_st = self.ride_fc(torch.cat(ride_reps, dim=1))

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

        output = dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
            state_embedding=ride_st.view(T, B, -1),
            int_baseline=self.int_baseline(core_output).view(T, B),
        )
        return (output, core_state)
