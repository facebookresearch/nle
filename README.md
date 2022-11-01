![NetHack Learning Environment (NLE)](https://github.com/facebookresearch/nle/raw/main/dat/nle/logo.png)

--------------------------------------------------------------------------------

<p align="center">
  <a href="https://circleci.com/gh/facebookresearch/nle">
    <img src="https://circleci.com/gh/facebookresearch/nle.svg?style=shield" />
  </a>
  <a href="https://github.com/facebookresearch/nle/actions/workflows/build_docker.yml">
    <img src="https://github.com/facebookresearch/nle/actions/workflows/test_and_deploy.yml/badge.svg?branch=main" />
  </a>
  <a href="https://pypi.python.org/pypi/nle/">
    <img src="https://img.shields.io/pypi/v/nle.svg" />
  </a>
   <a href="https://pepy.tech/project/nle">
    <img src="https://static.pepy.tech/personalized-badge/nle?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads" />
  </a>
   <a href="https://twitter.com/NetHack_LE">
    <img src="https://img.shields.io/twitter/follow/NetHack_LE?label=Twitter&style=social" alt="Twitter" />
  </a>
 </p>

The NetHack Learning Environment (NLE) is a Reinforcement Learning environment presented at [NeurIPS 2020](https://neurips.cc/Conferences/2020).
NLE is based on [NetHack 3.6.6](https://github.com/NetHack/NetHack/tree/NetHack-3.6.6_PostRelease) and designed to provide a standard RL interface to the game, and comes with tasks that function as a first step to evaluate agents on this new environment.

NetHack is one of the oldest and arguably most impactful videogames in history,
as well as being one of the hardest roguelikes currently being played by humans.
It is procedurally generated, rich in entities and dynamics, and overall an
extremely challenging environment for current state-of-the-art RL agents, while
being much cheaper to run compared to other challenging testbeds. Through NLE,
we wish to establish NetHack as one of the next challenges for research in
decision making and machine learning.

You can read more about NLE in the [NeurIPS 2020 paper](https://arxiv.org/abs/2006.13760), and about NetHack in its [original
README](./README.nh), at [nethack.org](https://nethack.org/), and on the
[NetHack wiki](https://nethackwiki.com).

![Example of an agent running on NLE](https://github.com/facebookresearch/nle/raw/main/dat/nle/example_run.gif)

### NLE Language Wrapper

We thank [ngoodger](https://github.com/ngoodger) for implementing the [NLE Language Wrapper](https://github.com/ngoodger/nle-language-wrapper) that translates the non-language observations from NetHack tasks into similar language representations. Actions can also be optionally provided in text form which are converted to the Discrete actions of the NLE.

### NetHack Learning Dataset

The NetHack Learning Dataset (NLD) code now ships with `NLE`, allowing users to the load large-scale datasets featured in [Dungeons and Data: A Large-Scale NetHack Dataset](), while also generating and loading their own datasets.

```python
import nle.dataset as nld

if not nld.db.exists():
    nld.db.create()
    # NB: Different methods are used for data based on NLE and data from NAO.
    nld.add_nledata_directory("/path/to/nld-aa", "nld-aa-v0")
    nld.add_altorg_directory("/path/to/nld-nao", "nld-nao-v0")

dataset = nld.TtyrecDataset("nld-aa-v0", batch_size=128, ...)
for i, mb in enumerate(dataset):
    foo(mb) # etc...
```

For information on how to download NLD-AA and NLD-NAO, see the dataset doc [here](./DATASET.md).

Otherwise checkout the tutorial Colab notebook [here](https://colab.research.google.com/drive/1GRP15SbOEDjbyhJGMDDb2rXAptRQztUD?usp=sharing).

# Papers using the NetHack Learning Environment
- Izumiya and Simo-Serra [Inventory Management with Attention-Based Meta Actions](https://esslab.jp/~ess/publications/IzumiyaCOG2021.pdf) (Waseda University, CoG 2021).
- Samvelyan et al. [MiniHack the Planet: A Sandbox for Open-Ended Reinforcement Learning Research](https://arxiv.org/abs/2109.13202) (FAIR, UCL, Oxford, NeurIPS 2021).
- Zhang et al. [BeBold: Exploration Beyond the Boundary of Explored Regions](https://arxiv.org/abs/2012.08621) (Berkley, FAIR, Dec 2020).
- Küttler et al. [The NetHack Learning Environment](https://arxiv.org/abs/2006.13760) (FAIR, Oxford, NYU, Imperial, UCL, NeurIPS 2020).

Open a [pull
request](https://github.com/facebookresearch/nle/edit/main/README.md)
to add papers.

# Getting started

Starting with NLE environments is extremely simple, provided one is familiar
with other gym / RL environments.


## Installation

NLE requires `python>=3.5`, `cmake>=3.15` to be installed and available both when building the
package, and at runtime.

On **MacOS**, one can use `Homebrew` as follows:

``` bash
$ brew install cmake
```

On a plain **Ubuntu 18.04** distribution, `cmake` and other dependencies
can be installed by doing:

```bash
# Python and most build deps
$ sudo apt-get install -y build-essential autoconf libtool pkg-config \
    python3-dev python3-pip python3-numpy git flex bison libbz2-dev

# recent cmake version
$ wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
$ sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
$ sudo apt-get update && apt-get --allow-unauthenticated install -y \
    cmake \
    kitware-archive-keyring
```

Afterwards it's a matter of setting up your environment. We advise using a conda
environment for this:

```bash
$ conda create -y -n nle python=3.8
$ conda activate nle
$ pip install nle
```


NOTE: If you want to extend / develop NLE, please install the package as follows:

``` bash
$ git clone https://github.com/facebookresearch/nle --recursive
$ pip install -e ".[dev]"
$ pre-commit install
```


## Docker

We have provided some docker images. Please see the [relevant
README](docker/README.md).


## Trying it out

After installation, one can try out any of the provided tasks as follows:

```python
>>> import gym
>>> import nle
>>> env = gym.make("NetHackScore-v0")
>>> env.reset()  # each reset generates a new dungeon
>>> env.step(1)  # move agent '@' north
>>> env.render()
```

NLE also comes with a few scripts that allow to get some environment rollouts,
and play with the action space:

```bash
# Play NetHackStaircase-v0 as a human
$ python -m nle.scripts.play

# Use a random agent
$ python -m nle.scripts.play --mode random

# Play the full game using directly the NetHack internal interface
# (Useful for debugging outside of the gym environment)
$ python -m nle.scripts.play --env NetHackScore-v0 # works with random agent too

# See all the options
$ python -m nle.scripts.play --help
```

Note that `nle.scripts.play` can also be run with `nle-play`, if the package
has been properly installed.

Additionally, a [TorchBeast](https://github.com/facebookresearch/torchbeast)
agent is bundled in `nle.agent` together with a simple model to provide a
starting point for experiments:

``` bash
$ pip install "nle[agent]"
$ python -m nle.agent.agent --num_actors 80 --batch_size 32 --unroll_length 80 --learning_rate 0.0001 --entropy_cost 0.0001 --use_lstm --total_steps 1000000000
```

Plot the mean return over the last 100 episodes:
```bash
$ python -m nle.scripts.plot
```
```
                              averaged episode return

  140 +---------------------------------------------------------------------+
      |             +             +            ++-+ ++++++++++++++++++++++++|
      |             :             :          ++++++++||||||||||||||||||||||||
  120 |-+...........:.............:...+-+.++++|||||||||||||||||||||||||||||||
      |             :        +++++++++++++++||||||||||AAAAAAAAAAAAAAAAAAAAAA|
      |            +++++++++++++||||||||||||||AAAAAAAAAAAA|||||||||||||||||||
  100 |-+......+++++|+|||||||||||||||||||||||AA||||||||||||||||||||||||||||||
      |       +++|||||||||||||||AAAAAAAAAAAAAA|||||||||||+++++++++++++++++++|
      |    ++++|||||AAAAAAAAAAAAAA||||||||||||++++++++++++++-+:             |
   80 |-++++|||||AAAAAA|||||||||||||||||||||+++++-+...........:...........+-|
      | ++|||||AAA|||||||||||||||++++++++++++-+ :             :             |
   60 |++||AAAAA|||||+++++++++++++-+............:.............:...........+-|
      |++|AA||||++++++-|-+        :             :             :             |
      |+|AA|||+++-+ :             :             :             :             |
   40 |+|A+++++-+...:.............:.............:.............:...........+-|
      |+AA+-+       :             :             :             :             |
      |AA-+         :             :             :             :             |
   20 |AA-+.........:.............:.............:.............:...........+-|
      |++-+         :             :             :             :             |
      |+-+          :             :             :             :             |
    0 |-+...........:.............:.............:.............:...........+-|
      |+            :             :             :             :             |
      |+            +             +             +             +             |
  -20 +---------------------------------------------------------------------+
      0           2e+08         4e+08         6e+08         8e+08         1e+09
                                       steps
```



# Contributing

We welcome contributions to NLE. If you are interested in contributing please
see [this document](./CONTRIBUTING.md).


# Architecture

NLE is direct fork of [NetHack](https://github.com/nethack/nethack) and
therefore contains code that operates on many different levels of abstraction.
This ranges from low-level game logic, to the higher-level administration of
repeated nethack games, and finally to binding of these games to Python `gym`
environment.

If you want to learn more about the architecture of `nle` and how it works
under the hood, checkout the [architecture document](./doc/nle/ARCHITECTURE.md).
This may be a useful starting point for anyone looking to contribute to the
lower level elements of NLE.


# Related Environments
- [gym\_nethack](http://campbelljc.com/research/gym_nethack/)
- [rogueinabox](https://github.com/rogueinabox/rogueinabox)
- [rogue-gym](https://github.com/kngwyu/rogue-gym)
- [MiniGrid](https://github.com/maximecb/gym-minigrid)
- [CoinRun](https://github.com/openai/coinrun)
- [MineRL](http://minerl.io/docs)
- [Project Malmo](https://www.microsoft.com/en-us/research/project/project-malmo/)
- [OpenAI Procgen Benchmark](https://openai.com/blog/procgen-benchmark/)
- [Obstacle Tower](https://github.com/Unity-Technologies/obstacle-tower-env)

# Interview about the environment with Weights&Biases
[Facebook AI Research’s Tim & Heiner on democratizing reinforcement learning research.](https://www.youtube.com/watch?v=oYSNXTkeCtw)

[![Interview with Weigths&Biases](https://img.youtube.com/vi/oYSNXTkeCtw/0.jpg)](https://www.youtube.com/watch?v=oYSNXTkeCtw)

# Citation

If you use NLE in any of your work, please cite:

```
@inproceedings{kuettler2020nethack,
  author    = {Heinrich K{\"{u}}ttler and
               Nantas Nardelli and
               Alexander H. Miller and
               Roberta Raileanu and
               Marco Selvatici and
               Edward Grefenstette and
               Tim Rockt{\"{a}}schel},
  title     = {{The NetHack Learning Environment}},
  booktitle = {Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2020},
}
```

If you use NLD or the datasets in any of your work, please cite:

```
@inproceedings{hambro2022dungeonsanddata,
  author    = {Eric Hambro and
               Roberta Raileanu and
               Danielle Rothermel and
               Vegard Mella and
               Tim Rockt{\"{a}}schel and
               Heinrich K{\"{u}}ttler and
               Naila Murray},
  title     = {{Dungeons and Data: A Large-Scale NetHack Dataset}},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2022},
  url       = {https://openreview.net/forum?id=zHNNSzo10xN}
}
```