



# MiniHack

The MiniHack is a Reinforcement Learning environment which extends the NetHack Learning Environment (NLE) presented at [NeurIPS 2020](https://neurips.cc/Conferences/2020).
Similar to NLE, MiniHack is based on [NetHack 3.6.6](https://github.com/NetHack/NetHack/tree/NetHack-3.6.6_PostRelease) and designed to provide a standard RL interface to the game, and comes with tasks that function as a first step to evaluate agents on this new environment.

# Getting started

Starting with NLE environments is extremely simple, provided one is familiar with other gym / RL environments.


## Installation

Please take a look at [here](https://github.com/facebookresearch/nle/blob/master/README.md) for detailed instructions on how to install NLE.


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
see [this document](./CONTRIBUTING.md) 

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