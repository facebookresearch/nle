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

To use:
1) forward a port to 5005

2) start hiplot

`python -m hackrl.scripts.hiplot`

3) open hiplot in the browser at localhost:5000 and enter the {sweep_path} with globbing

e.g.
`/home/user/outputs/2020-07-24/07-04-02/*/logs.csv`

This collects logs via a function imported from the gnuplot plot script.

"""

import copy

import hiplot as hip

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from threading import Timer
from nle.agent.scripts.plot import collect_logs


# Default hiplot server.
HIPLOT_SERVER_URL = "http://127.0.0.1:5005/"


def flatten(cfg):
    """Collapse configurations -- {"foo": {"bar": 0}} -> {"foo.bar": 0}"""
    flat = False
    while not flat:
        flat = True
        new_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, DictConfig) or isinstance(val, dict):
                flat = False
                for subkey, subval in val.items():
                    newkey = key + "." + subkey
                    new_cfg[newkey] = subval
            else:
                new_cfg[key] = val
        cfg = new_cfg
    return new_cfg


def fetcher(uri):
    """Prepare param sweep output for hiplot
    Collects the sweep results and simplifies them for easy display using hiplot.
    :param uri: root dir that containing all the param_sweeping results.
    :returns: hiplot Experiment Object for display
    """

    print("got request for %s, collecting logs" % uri)

    exp = hip.Experiment()
    exp.display_data(hip.Displays.XY).update(
        {"axis_x": "step", "axis_y": "cumulative_reward"}
    )

    dfs = collect_logs(Path(uri))  # list of (name, log, df) triplets
    cfg_variants = {}
    cfgs = {}
    for name, _dfs in dfs:
        # first collect each config
        print("loading config from %s" % name)
        target = Path(name)
        configpath = target / "config.yaml"
        cfg = flatten(OmegaConf.load(str(configpath)))
        cfgs[name] = cfg
        for k, v in cfg.items():
            if k not in cfg_variants:
                cfg_variants[k] = set()
            cfg_variants[k].add(v)

    print("Read in %d logs successfully" % len(cfgs))

    order = []
    order.append("mean_final_reward")
    # cfg_variants are hyperparams with more than one value
    for key, vals in cfg_variants.items():
        if len(vals) > 1:
            order.append(key)
    order.append("cumulative_reward")
    print("headers found to plot: ", order)
    exp.display_data(hip.Displays.PARALLEL_PLOT).update(
        hide=["step", "uid", "from_uid"], order=order
    )

    # min_points = min(len(df["step"]) for _name, df in dfs)
    # max_points = max(len(df["step"]) for _name, df in dfs)
    ave_points = sum(len(df["step"]) for _name, df in dfs) // len(dfs)
    step_size = ave_points // 100 + 1  # I want an average of 100 points per experiment
    print("ave_points:", ave_points, "step_size:", step_size)

    for name, df in dfs:
        # now go through each dataframe
        cfg = cfgs[name]

        hyperparams = dict()
        for key, val in cfg.items():
            if len(cfg_variants[key]) > 1:
                try:
                    hyperparams[key] = float(val)
                except ValueError:
                    hyperparams[key] = str(val)

        steps = df["step"]
        prev_name = None
        cum_sum = df["mean_episode_return"].cumsum()

        for idx in range(0, len(cum_sum), step_size):
            step = int(steps[idx])
            cumulative_reward = cum_sum[idx]
            curr_name = "{},step{}".format(name, step)
            sp = hip.Datapoint(
                uid=curr_name,
                values=dict(step=step, cumulative_reward=cumulative_reward),
            )
            if prev_name is not None:
                sp.from_uid = prev_name
            exp.datapoints.append(sp)
            prev_name = curr_name

        mean_final_reward = float(df["mean_episode_return"][-10000:].mean())
        peak_performance = float(
            df["mean_episode_return"].rolling(window=1000).mean().max()
        )
        end_vals = copy.deepcopy(hyperparams)
        end_vals.update(
            step=int(steps.iloc[-1]),
            cumulative_reward=cum_sum.iloc[-1],
            mean_final_reward=mean_final_reward,
            peak_performance=peak_performance,
        )
        dp = hip.Datapoint(uid=name, from_uid=prev_name, values=end_vals)
        exp.datapoints.append(dp)

    return exp


def open_browser():
    import webbrowser

    webbrowser.open(HIPLOT_SERVER_URL, new=2, autoraise=True)


def main():
    # By running the following command, a hiplot server will be rendered to display
    # your experiment results using the udf fetcher passed to hiplot.
    try:
        Timer(1, open_browser).start()
    except Exception as e:
        print("Fail to open browser", e)
    hip.server.run_server(fetchers=[fetcher])


if __name__ == "__main__":
    main()
