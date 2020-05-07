![NetHack Learning Environment (NLE)](dat/nle/logo.png)

--------------------------------------------------------------------------------

The NetHack Learning Environment (NLE) is a Reinforcement Learning environment
based on [NetHack 3.6.6](https://github.com/NetHack/NetHack/tree/NetHack-3.6.6_PostRelease).
NLE is designed to provide a standard RL interface to the game, and comes with
tasks that function as a first step to evaluate agents on this new environment.

NetHack is one of the oldest and arguably most impactful videogames in history,
as well as being one of the hardest roguelikes currently being played by humans.
It is procedurally generated, rich in entities and dynamics, and overall an
extremely challing environment for current state-of-the-art RL agents, while
being much cheaper to run compared to other challenging testbeds. Through NLE,
we wish to establish NetHack as one of the next challenges for research in
decision making and machine learning.

You can read more about NetHack in its [original README](./README.nh), at
[nethack.org](https://nethack.org/), and on the [NetHack
wiki](https://nethackwiki.com).

![Example of an agent running on NLE](dat/nle/example_run.gif)


# Getting started

Starting with NLE environments is extremely simple, provided one is familiar
with other gym / RL environments.


## Installation

NLE requires `python>=3.7`, `libzmq`, `flatbuffers`, and some NetHack
dependencies (e.g. `libncurses`) to be installed and available both when
building the package, and at runtime.


On **MacOS**, one can use `Homebrew` as follows:

``` bash
$ brew install ncurses flatbuffers zeromq
$ sudo wget https://raw.githubusercontent.com/zeromq/cppzmq/v4.3.0/zmq.hpp -P \
     /usr/local/include
```

On a plain **Ubuntu 18.04** distribution, `flatbuffers` and other dependencies
can be installed by doing:

```bash
# zmq, python, and build deps
$ sudo apt-get install -y build-essential autoconf libtool pkg-config \
    python3-dev python3-pip python3-numpy git cmake libncurses5-dev \
    libzmq3-dev flex bison
# building flatbuffers
$ git clone https://github.com/google/flatbuffers.git
$ cd flatbuffers
$ cmake -G "Unix Makefiles"
$ make
$ sudo make install
```

Afterwards it's a matter of setting up your environment. We advise using a conda
environment for this:

```bash
$ conda create -n nle python=3.8
$ conda activate nle
$ conda install cppzmq  # might not be necessary on some systems
$ pip install nle
```


NOTE: If you want to extend / develop NLE, please install the package as follows:

``` bash
$ git clone git@github.com:facebookresearch/nle
$ pip install -e ".[dev]"
$ pre-commit install
```


## Docker

We have provided some docker images. Please see the [relevant README](docker/README.md).


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
$ python -m nle.scripts.play --env nethack  # works with random agent too

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


# Related Environments
- [gym\_nethack](http://campbelljc.com/research/gym_nethack/)
- [rogueinabox](https://github.com/rogueinabox/rogueinabox)
- [rogue-gym](https://github.com/kngwyu/rogue-gym)
- [MiniGrid](https://github.com/maximecb/gym-minigrid)
- [CoinRun](https://github.com/openai/coinrun)
- [Project Malmo](https://www.microsoft.com/en-us/research/project/project-malmo/)
- [OpenAI Procgen Benchmark](https://openai.com/blog/procgen-benchmark/)
- [Obstacle Tower](https://github.com/Unity-Technologies/obstacle-tower-env)


# Citation

If you use NLE in any of your work, please cite:

```
@inproceedings{kuettler2020nethack,
  title={{The NetHack Learning Environment}},
  author={Heinrich K\"{u}ttler and Nantas Nardelli and Roberta Raileanu and Marco Selvatici and Edward Grefenstette and Tim Rockt\"{a}schel},
  year={2020},
  booktitle={Workshop on Beyond Tabula Rasa in Reinforcement Learning (BeTR-RL)},
  url={https://github.com/facebookresearch/nle},
}
```
