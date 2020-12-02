# Neurips 2020 code release

Here we release updated code to get results competitive with our NeurIPS 2020 paper.

To be clear, this is not the exact code used for the paper: we made a number of performance improvements to NLE since the original results, dramatically increasing the speed of the environment (which was already one of the fastest-performing environments when the paper was published!).

We also introduced some additional modeling options, including conditioning the model on the in-game messages (i.e. msg.model=lt_cnn) and introducing new ways of observing the environment through different glyph types (i.e. glyph_type=all_cat). These features are enabled by default for the model now, which outperforms the models in the paper.

## Reproduced results

After 1e9 training steps, the average mean_episode_return achieved by the agents in their last 10k episodes are listed below, averaged over three runs.

We give 1 reward for winning the staircase, pet, and oracle tasks. This gives a lower reward per completion than the original paper.
We give 1 reward for every 1000 steps the agent stays alive on all tasks. This is absent in the original paper.

**Staircase Task**
The average steps per episode for all models was about 400, so mean_episode_return - 0.4 gives the approximate percentage success rate of the task (so getting close to 1.4 means the agent is reaching the goal every episode).
These models perform better than the original paper result.
- mon-hum-neu-mal: baseline 1.37, RND 1.00. 
- val-dwa-law-fem: baseline 1.17, RND 1.14
- wiz-elf-cha-mal: baseline 1.01, RND 0.97
- tou-hum-neu-fem: baseline 0.94, RND 1.18

**Pet Task**
The average steps per episode for all models was about 400, so mean_episode_return - 0.4 gives the approximate percentage success rate of the task (so getting close to 1.4 means the agent is reaching the goal every episode).
These models perform better than the original paper result.
- mon-hum-neu-mal: baseline 1.26, RND 1.31
- val-dwa-law-fem: baseline 1.08, RND 1.08
- wiz-elf-cha-mal: baseline 0.86, RND 0.85
- tou-hum-neu-fem: baseline 0.75, RND 0.79

**Eat Task**
The valkyrie baseline performed worse than in the original paper on this task; otherwise, every other model and character performed much better.
- mon-hum-neu-mal: baseline 2282, RND 2193
- val-dwa-law-fem: baseline 145,  RND 1240
- wiz-elf-cha-mal: baseline 993,  RND 1066
- tou-hum-neu-fem: baseline 1131, RND 1230

**Gold Task**
These models perform much better than the original paper result.
- mon-hum-neu-mal: baseline 159, RND 116
- val-dwa-law-fem: baseline 60,  RND 63
- wiz-elf-cha-mal: baseline 37, RND 38
- tou-hum-neu-fem: baseline 18, RND 19

**Score Task**
The RND models here perform slightly worse than in the paper.
The baseline models here perform significantly better than in the paper.
- mon-hum-neu-mal: baseline 971, RND 941
- val-dwa-law-fem: baseline 653, RND 657
- wiz-elf-cha-mal: baseline 343, RND 343
- tou-hum-neu-fem: baseline 116, RND 202

**Scout Task**
These models perform significantly better than in the paper.
- mon-hum-neu-mal: baseline 2491, RND 2452
- val-dwa-law-fem: baseline 1978, RND 1982
- wiz-elf-cha-mal: baseline 1411, RND 1397
- tou-hum-neu-fem: baseline 1112, RND 1114

**Oracle Task**
These models perform similarly to the paper.
- mon-hum-neu-mal: baseline -4.1, RND -4.8
- val-dwa-law-fem: baseline -4.9, RND -6.0
- wiz-elf-cha-mal: baseline -4.4, RND -4.3
- tou-hum-neu-fem: baseline -3.4, RND -6.8


## Changed params from the paper (better performing!)

- change the reward_win parameter from 100 to 1. this only affects the staircase, pet, and oracle tasks. this is a more appropriate ratio between the reward and the step penalty and results in more consistent performance. you can compare with the plots in the paper but consider the scale to be leading towards 1 rather than 100 for the agent to be reaching the goal on every episode.
- decrease learning rate from 0.0002 to 0.0001
- added reward_normalization parameter. set this reward_normalization=true, set reward_clipping=none (was "tim" before).
- increase hidden_size from 128 to 256
- increase embedding size from 32 to 64
- add a "message model" which conditions on the in-game message, providing a fourth input to the policy (in addition to the full dungeon screen, the crop of the dungeon right around the agent, and the status bar). set msg.model=lt_cnn.
- add different interpretations of the glyphs in the environment. see below for explanation. set glyph_type=all_cat.

When msg.model=lt_cnn, and int.input=full (the default), we also add the message model to the target and predictor networks for RND. This should also improve RND network's performance, as seeing new messages in the game should be a particularly high-signal new experience to seek out (it could include taking new actions or seeing new monsters, new items, new environments or more) but we have not yet analyzed this in detail.

## Glyph Types

The default setting `full` from in the paper uses a unique identifier for every glyph (entity on the screen) that you encounter. There are about 6000 unique glyphs.

However, this masks the relationships between entities that the agent might otherwise be able to use (and humans do!), for example that several entities that share certain traits might have a similar appearance such as different kinds of dogs using the same symbol but having different colours.

The `group_id` setting is one such breakdown, splitting each entity into one of twelve groups and then an id within each group.

An alternative breakdown `color_char`(_special) is by actual appearance, splitting each glyph into the colour used for the entity, the character used for it, and then finally a special identifier which plays can enable in the game and provides additional information about the entity at that tile.

We then also provide the `all` encoding which uses the group, id, colour, character, and special traits for each item, embedding them each in an `edim` vector, concatenating the vectors, and then projecting back to a final `edim` vector using a Linear(5 * edim, edim) layer.

Finally, we provide `all_cat` which partitions the `edim` final vector into sub-vectors based on how unique each component is so that the final vectors can be concatened (without projection) into an `edim` vector. That is, for a 64-dim embedding_dim, we would use a 24-dim embedding for the id, 8 dim for group, 8 dim for color, 16 dim for character, and 8 dim for special. This is a bit arbitrary but performs well and is the recommended setting.

## Polyhydra Syntax

```
# install hydra
pip install hydra-core hydra_colorlog

# single run
python -m hackrl.polyhydra model=baseline env=score num_actors=80

# to sweep on the cluster: add another -m (for multirun) and comma-separate values
python -m hackrl.polyhydra model=baseline,ride,rnd,dynamics env=score,gold

# hydra supports nested arguments.
python -m hackrl.polyhydra msg.model=cnn fwd.forward_cost=0.01 ride.count_norm=false
```

In addition to specifiying arguments on the command line, you can edit config.yaml directly.

## Reproducing our NeurIPS sweep

Run neurips_sweep.sh to run a sweep covering the characters, tasks, and models we report in the paper.