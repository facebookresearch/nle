#!/usr/bin/env python
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
Script for plotting results from an NLE agent's logs.tsv file.

Some examples of using the plotting tool:

Plot the most recent run (symlinked at ~/torchbeast/latest by default).
```
python -m nle.scripts.plot
```

Plot a specific run using a rolling window of size 10 (if window > 1, shows error bars).
```
python -m nle.scripts.plot path/to/run/logs.tsv --window 10
```

Plot a specific run to a specific window size. (PATH/logs.tsv is found automatically)
```
python -m nle.scripts.plot path/to/run -x 100 -y 50
```

Plot all runs under a specific directory without a legend matching plots to runs.
```
python -m nle.scripts.plot path/to/multiple_runs --no_legend
```

Plot all runs matching a directory prefix, zooming in on a specific prefix.
Note that negative ranges need a little help on the command line.
```
python -m nle.scripts.plot path/to/multiple_runs/2020 --xrange 0,1e8 --yrange='-10,80'
```
"""

import argparse
import gnuplotlib as gp
import numpy as np
import pandas as pd
import random

from pathlib import Path


def str_to_float_pair(s):
    """
    Convert string to pair of floats.
    """
    if s is None:
        return None
    split = s.split(",")
    if len(split) != 2:
        raise RuntimeError("range does not match pattern 'float,float'")
    return (float(split[0]), float(split[1]))


parser = argparse.ArgumentParser("NetHack GnuPlotter", allow_abbrev=False)
parser.register("type", "pair", str_to_float_pair)
parser.add_argument(
    "-f",
    "--file",
    type=str,
    default="~/torchbeast/latest/logs.tsv",
    help="file to plot or directory to look for log files",
)
parser.add_argument(
    "-w", "--window", type=int, default=-1, help="override automatic window size."
)
parser.add_argument("-x", "--width", type=int, default=80, help="width of plot")
parser.add_argument("-y", "--height", type=int, default=30, help="height of plot")
parser.add_argument(
    "--no_legend",
    action="store_true",
    help="skip printing legend when plotting multiple experiments",
)
parser.add_argument(
    "--xrange",
    type="pair",
    default=None,
    help="float,float. range of x values to plot. overrides automatic zoom for x axis.",
)
parser.add_argument(
    "--yrange",
    type="pair",
    default=None,
    help="float,float. range of y values to plot. overrides automatic zoom for y axis.",
)
parser.add_argument(
    "--shuffle",
    action="store_true",
    help="shuffles the order of plotting if rendering multiple curves.",
)


def plot_single_ascii(target, width, height, window=-1, xrange=None, yrange=None):
    """
    Plot the target file using the specified width and height.
    If window > 0, use it to specify the window size for rolling averages.
    xrange and yrange are used to specify the zoom level of the plot.
    """
    print("plotting %s" % str(target))
    df = pd.read_csv(target, sep="\t")
    steps = np.array(df["# Step"])

    if window < 0:
        window = len(steps) // width + 1
    window = df["mean_episode_return"].rolling(window=window, min_periods=0)
    returns = np.array(window.mean())
    stderrs = np.array(window.std())

    plot_options = {}
    plot_options["with"] = "yerrorbars"
    plot_options["terminal"] = "dumb %d %d ansi" % (width, height)
    plot_options["tuplesize"] = 3
    plot_options["title"] = "averaged episode return"
    plot_options["xlabel"] = "steps"

    if xrange is not None:
        plot_options["xrange"] = xrange

    if yrange is not None:
        plot_options["yrange"] = yrange

    gp.plot(steps, returns, stderrs, **plot_options)


def collect_logs(target):
    """
    Collect results from log files at the target directory.
    Can be fully specified or a partial match, for example:
        full: /checkpoint/me/outputs/2020-05-12/00-02-13/
        part: /checkpoint/me/outputs/2020-05-12/00
    """
    dfs = []
    if target.is_dir():
        # fully specified directory
        glob = target.glob("**/logs.tsv")
    else:
        # partially specified directory, glob from parent dir
        glob = target.parent.glob("**/logs.tsv")
    for child in glob:
        if target.name in str(child.parent):
            # this allows users to provide partial dir names, e.g. "~/logs/2020-"
            try:
                df = pd.read_csv(child, sep="\t")
                # TODO: remove rows with nan? maybe bad as will damage rolling window
                # df[df["mean_episode_return"] == df["mean_episode_return"]].copy()
                if len(df) > 0:
                    name = str(child.parent)
                    dfs.append((name, df))
            except pd.errors.EmptyDataError:
                # nothing to plot
                pass

    if len(dfs) == 0:
        # didn't find any valid tsv logs
        raise FileNotFoundError("No logs found under %s" % target)

    return dfs


def plot_multiple_ascii(
    target,
    width,
    height,
    window=-1,
    xrange=None,
    yrange=None,
    no_legend=False,
    shuffle=False,
):
    """
    Plot files under the target path using the specified width and height.
    If window > 0, use it to specify the window size for rolling averages.
    xrange and yrange are used to specify the zoom level of the plot.
    Set no_legend to true to save the visual space for the plot.
    shuffle randomizes the order of the plot (does NOT preserve auto-assigned curve
        labels), which can help to see a curve which otherwise is overwritten.
    """
    dfs = collect_logs(target)

    if window < 0:
        max_size = max(len(df["# Step"]) for name, df in dfs)
        window = 2 * max_size // width + 1

    datasets = []
    for name, df in dfs:
        steps = np.array(df["# Step"])
        if window > 1:
            roll = df["mean_episode_return"].rolling(window=window, min_periods=0)
            try:
                rewards = np.array(roll.mean())
            except pd.core.base.DataError:
                print("Error reading file at %s" % name)
                continue
        else:
            rewards = np.array(df["mean_episode_return"])
        if no_legend:
            datasets.append((steps, rewards))
        else:
            datasets.append((steps, rewards, dict(legend="    " + name + ":")))

    errs = len(dfs) - len(datasets)
    if errs > 0:
        print(
            "Skipped %d runs (%f) due to errors reading data" % (errs, errs / len(dfs))
        )
    print(
        "Plotting %d runs with window_size %d from %s" % (len(datasets), window, target)
    )

    plot_options = {}
    plot_options["terminal"] = "dumb %d %d ansi" % (width, height)
    plot_options["tuplesize"] = 2
    plot_options["title"] = "averaged episode return"
    plot_options["xlabel"] = "steps"
    plot_options["set"] = "key outside below"

    if xrange is not None:
        plot_options["xrange"] = xrange

    if yrange is not None:
        plot_options["yrange"] = yrange

    if shuffle:
        random.shuffle(datasets)
    gp.plot(*datasets, **plot_options)


def plot(flags):
    target = Path(flags.file).expanduser()

    if target.is_file():
        # plot single torchbeast run, path/to/logs.tsv
        if target.suffix == ".tsv":
            plot_single_ascii(
                target,
                flags.width,
                flags.height,
                flags.window,
                flags.xrange,
                flags.yrange,
            )
        else:
            raise RuntimeError(
                "Filetype not recognised (expected .tsv): %s" % target.suffix
            )
    elif (target / "logs.tsv").is_file():
        # next check if this is actually a single run directory with file "logs.tsv"
        plot_single_ascii(
            target / "logs.tsv",
            flags.width,
            flags.height,
            flags.window,
            flags.xrange,
            flags.yrange,
        )
    else:
        # look for runs underneath the specified directory
        plot_multiple_ascii(
            target,
            flags.width,
            flags.height,
            flags.window,
            flags.xrange,
            flags.yrange,
            flags.no_legend,
            flags.shuffle,
        )


if __name__ == "__main__":
    flags = parser.parse_args()
    plot(flags)
