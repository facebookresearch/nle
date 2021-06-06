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

from nle.env import tasks
from nle.env.base import DUNGEON_SHAPE

from .baseline import BaselineNet

from omegaconf import OmegaConf
import torch


ENVS = dict(
    staircase=tasks.NetHackStaircase,
    score=tasks.NetHackScore,
    pet=tasks.NetHackStaircasePet,
    oracle=tasks.NetHackOracle,
    gold=tasks.NetHackGold,
    eat=tasks.NetHackEat,
    scout=tasks.NetHackScout,
    challenge=tasks.NetHackChallenge,
)


def create_model(flags, device):
    model_string = flags.model
    if model_string == "baseline":
        model_cls = BaselineNet
    else:
        raise NotImplementedError("model=%s" % model_string)

    action_space = ENVS[flags.env](savedir=None, archivefile=None)._actions

    model = model_cls(DUNGEON_SHAPE, action_space, flags, device)
    model.to(device=device)
    return model


def load_model(load_dir, device):
    flags = OmegaConf.load(load_dir + "/config.yaml")
    flags.checkpoint = load_dir + "/checkpoint.tar"
    model = create_model(flags, device)
    checkpoint_states = torch.load(flags.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_states["model_state_dict"])
    return model
