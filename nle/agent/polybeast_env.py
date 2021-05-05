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

import argparse
import multiprocessing as mp
import logging
import os
import threading
import time

from nle.agent.envs import tasks
import libtorchbeast


# yapf: disable
parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument('--env', default='staircase', type=str, metavar='E',
                    help='Name of Gym environment to create.')
parser.add_argument('--character', default='mon-hum-neu-mal', type=str, metavar='C',
                    help='Specification of the NetHack character.')
parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument('--num_servers', default=4, type=int, metavar='N',
                    help='Number of environment servers.')

parser.add_argument('--mock', action="store_true",
                    help='Use mock environment instead of NetHack.')
parser.add_argument('--single_ttyrec', action="store_true",
                    help='Record ttyrec only for actor 0.')
parser.add_argument('--num_seeds', default=0, type=int, metavar='S',
                    help='If larger than 0, samples fixed number of environment seeds '
                         'to be used.')
parser.add_argument('--seedspath', default="", type=str,
                    help="Path to json file with seeds.")

# Training settings.
parser.add_argument('--savedir', default='~/nethackruns',
                    help='Root dir where experiment data will be saved.')

# Task-Specific settings.
parser.add_argument('--reward_win', default=1.0, type=float,
                    help='Reward for winning (finding the staircase).')
parser.add_argument('--reward_lose', default=-1.0, type=float,
                    help='Reward for losing (dying before finding the staircase).')

parser.add_argument('--penalty_step', default=-0.0001, type=float,
                    help='Penalty per step in the episode.')
parser.add_argument('--penalty_time', default=-0.0001, type=float,
                    help='Penalty per time step in the episode.')
parser.add_argument('--fn_penalty_step', default="constant", type=str,
                    help='Function to accumulate penalty.')
parser.add_argument('--max_num_steps', default=1000, type=int,
                    help='Maximum number of steps in the game.')
parser.add_argument('--state_counter', default="none", choices=['none', 'coordinates'],
                    help='Method for counting state visits. Default none. '
                         'Coordinates concatenates dungeon level with player x,y.')
# yapf: enable

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def create_folders(flags):
    # Creates some of the folders that would be created by the filewriter.
    logdir = os.path.join(flags.savedir, "archives")
    if not os.path.exists(logdir):
        logging.info("Creating archive directory: %s" % logdir)
        os.makedirs(logdir, exist_ok=True)
    else:
        logging.info("Found archive directory: %s" % logdir)


def create_env(flags, env_id=0, lock=threading.Lock()):
    # Create environment instances for actors
    with lock:
        env_class = tasks.ENVS[flags.env]
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
            ),
            penalty_step=flags.penalty_step,
            penalty_time=flags.penalty_time,
            penalty_mode=flags.fn_penalty_step,
        )
        if flags.env in ("staircase", "pet", "oracle") or any(
            name in flags.env for name in ("room", "corridor", "keyroom")
        ):  # TODO MIKA FIX
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


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
