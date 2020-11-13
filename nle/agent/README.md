# Neurips 2020 code release

Here we release updated code to get results competitive with our NeurIPS 2020 paper.

To be clear, this is not the exact code used for the paper: we made a number of performance improvements to NLE since the original results, dramatically increasing the speed of the environment (which was already one of the fastest-performing environments when the paper was published!).

We also introduced some additional modeling options, including conditioning the model on the in-game messages (i.e. msg.model=lt_cnn) and introducing new ways of observing the environment through different glyph types (i.e. glyph_type=all_cat). These features are enabled by default for the model now, which outperforms the models in the paper.

# Polyhydra Syntax

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

# Reproducing our NeurIPS sweep

Run neurips_sweep.sh to run a sweep covering the characters, tasks, and models we report in the paper.