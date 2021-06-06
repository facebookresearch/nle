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

"""
Installation for hydra:
pip install hydra-core hydra_colorlog --upgrade

Runs like polybeast but use = to set flags:
python -m polyhydra.py learning_rate=0.001 rnd.twoheaded=true

Run sweep with another -m after the module:
python -m polyhydra.py -m learning_rate=0.01,0.001,0.0001,0.00001 momentum=0,0.5

Baseline should run with:
python polyhydra.py
"""

from pathlib import Path
import logging
import os
import multiprocessing as mp

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig

import torch

import polybeast_env
import polybeast_learner

if torch.__version__.startswith("1.5") or torch.__version__.startswith("1.6"):
    # pytorch 1.5.* needs this for some reason on the cluster
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def pipes_basename():
    logdir = Path(os.getcwd())
    name = ".".join([logdir.parents[1].name, logdir.parents[0].name, logdir.name])
    return "unix:/tmp/poly.%s" % name


def get_common_flags(flags):
    flags = OmegaConf.to_container(flags)
    flags["pipes_basename"] = pipes_basename()
    flags["savedir"] = os.getcwd()
    return OmegaConf.create(flags)


def get_learner_flags(flags):
    lrn_flags = OmegaConf.to_container(flags)
    lrn_flags["checkpoint"] = os.path.join(flags["savedir"], "checkpoint.tar")
    lrn_flags["entropy_cost"] = float(lrn_flags["entropy_cost"])
    return OmegaConf.create(lrn_flags)


def run_learner(flags: DictConfig):
    polybeast_learner.main(flags)


def get_environment_flags(flags):
    env_flags = OmegaConf.to_container(flags)
    env_flags["num_servers"] = flags.num_actors
    max_num_steps = 1e6
    if flags.env in ("staircase", "pet"):
        max_num_steps = 1000
    env_flags["max_num_steps"] = int(max_num_steps)
    env_flags["seedspath"] = ""
    return OmegaConf.create(env_flags)


def run_env(flags):
    np.random.seed()  # Get new random seed in forked process.
    polybeast_env.main(flags)


def symlink_latest(savedir, symlink):
    try:
        if os.path.islink(symlink):
            os.remove(symlink)
        if not os.path.exists(symlink):
            os.symlink(savedir, symlink)
            logging.info("Symlinked log directory: %s" % symlink)
    except OSError:
        # os.remove() or os.symlink() raced. Don't do anything.
        pass


@hydra.main(config_name="config")
def main(flags: DictConfig):
    if os.path.exists("config.yaml"):
        # this ignores the local config.yaml and replaces it completely with saved one
        logging.info("loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load("config.yaml")
        cli_conf = OmegaConf.from_cli()
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        flags = OmegaConf.merge(new_flags, cli_conf)
    if flags.load_dir and os.path.exists(os.path.join(flags.load_dir, "config.yaml")):
        new_flags = OmegaConf.load(os.path.join(flags.load_dir, "config.yaml"))
        cli_conf = OmegaConf.from_cli()
        flags = OmegaConf.merge(new_flags, cli_conf)

    logging.info(flags.pretty(resolve=True))
    OmegaConf.save(flags, "config.yaml")

    flags = get_common_flags(flags)

    # set flags for polybeast_env
    env_flags = get_environment_flags(flags)
    env_processes = []
    for _ in range(1):
        p = mp.Process(target=run_env, args=(env_flags,))
        p.start()
        env_processes.append(p)

    symlink_latest(
        flags.savedir, os.path.join(hydra.utils.get_original_cwd(), "latest")
    )

    lrn_flags = get_learner_flags(flags)
    run_learner(lrn_flags)

    for p in env_processes:
        p.kill()
        p.join()


if __name__ == "__main__":
    main()
