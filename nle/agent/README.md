# Running Baselines

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

## Reproducing our NeurIPS results

Please take a look into NeurIPS2020.md for details on how to reproduce results from the NLE paper as published at NeurIPS 2020.