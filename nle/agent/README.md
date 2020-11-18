# Neurips 2020 code release

Here we release updated code to get results competitive with our NeurIPS 2020 paper.

To be clear, this is not the exact code used for the paper: we made a number of performance improvements to NLE since the original results, dramatically increasing the speed of the environment (which was already one of the fastest-performing environments when the paper was published!).

We also introduced some additional modeling options, including conditioning the model on the in-game messages (i.e. msg.model=lt_cnn) and introducing new ways of observing the environment through different glyph types (i.e. glyph_type=all_cat). These features are enabled by default for the model now, which outperforms the models in the paper.

## Reproduced results

We are still rerunning experiments using the below params and will be happy to report less variability as well as higher returns compared to the results reported in the paper.

## Changed params from the paper (better performing!)

- change the reward_win parameter from 100 to 1. this only affects the staircase, pet, and oracle tasks. this is a more appropriate ratio between the reward and the step penalty and results in more consistent performance. you can compare with the plots in the paper directly 
- added reward_normalization parameter. set this reward_normalization=true, set reward_clipping=none (was "tim" before).
- increase hidden_size from 128 to 256
- increase embedding size from 32 to 64
- add a "message model" which conditions on the in-game message, providing a fourth input to the policy (in addition to the full dungeon screen, the crop of the dungeon right around the agent, and the status bar). set msg.model=lt_cnn.
- add different interpretations of the glyphs in the environment. see below for explanation. set glyph_type=all_cat.

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