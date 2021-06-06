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

import multiprocessing as mp
import logging
import os
import threading
import time

import torch

import libtorchbeast

from models import ENVS


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


# Helper functions for NethackEnv.
def _format_observation(obs):
    obs = torch.from_numpy(obs)
    return obs.view((1, 1) + obs.shape)  # (...) -> (T,B,...).


def create_folders(flags):
    # Creates some of the folders that would be created by the filewriter.
    logdir = os.path.join(flags.savedir, "archives")
    if not os.path.exists(logdir):
        logging.info("Creating archive directory: %s" % logdir)
        os.makedirs(logdir, exist_ok=True)
    else:
        logging.info("Found archive directory: %s" % logdir)


def create_env(flags, env_id=0, lock=threading.Lock()):
    # commenting out these options for now because they use too much disk space
    # archivefile = "nethack.%i.%%(pid)i.%%(time)s.zip" % env_id
    # if flags.single_ttyrec and env_id != 0:
    #     archivefile = None

    # logdir = os.path.join(flags.savedir, "archives")

    with lock:
        env_class = ENVS[flags.env]
        kwargs = dict(
            savedir=None,
            archivefile=None,
            character=flags.character,
            max_episode_steps=flags.max_num_steps,
            observation_keys=(
                "glyphs",
                "chars",
                "colors",
                "specials",
                "blstats",
                "message",
                "tty_chars",
                "tty_colors",
                "tty_cursor",
                "inv_glyphs",
                "inv_strs",
                "inv_letters",
                "inv_oclasses",
            ),
            penalty_step=flags.penalty_step,
            penalty_time=flags.penalty_time,
            penalty_mode=flags.fn_penalty_step,
        )
        if flags.env in ("staircase", "pet", "oracle"):
            kwargs.update(reward_win=flags.reward_win, reward_lose=flags.reward_lose)
        elif env_id == 0:  # print warning once
            print("Ignoring flags.reward_win and flags.reward_lose")
        if flags.state_counter != "none":
            kwargs.update(state_counter=flags.state_counter)
        env = env_class(**kwargs)
        if flags.seedspath is not None and len(flags.seedspath) > 0:
            raise NotImplementedError("seedspath > 0 not implemented yet.")

        return env


def serve(flags, server_address, env_id):
    env = lambda: create_env(flags, env_id)
    server = libtorchbeast.Server(env, server_address=server_address)
    server.run()


def main(flags):
    if flags.num_seeds > 0:
        raise NotImplementedError("num_seeds > 0 not currently implemented.")

    create_folders(flags)

    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    processes = []
    for i in range(flags.num_servers):
        p = mp.Process(
            target=serve, args=(flags, f"{flags.pipes_basename}.{i}", i), daemon=True
        )
        p.start()
        processes.append(p)

    try:
        # We are only here to listen to the interrupt.
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
