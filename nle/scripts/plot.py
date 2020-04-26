import pandas as pd
import matplotlib.pyplot as plt
from absl import app
from absl import flags
import termplotlib as tpl


FLAGS = flags.FLAGS

flags.DEFINE_string("path", "~/torchbeast/latest/logs.tsv", "Path to logs file.")
flags.DEFINE_string("out", "output.png", "Path to output plot file.")
flags.DEFINE_string("metric", "mean_episode_return", "Metric to plot.")
flags.DEFINE_integer("window", 100, "Number of episode to average over.")
flags.DEFINE_bool("ascii", True, "Plot using termplotlib.")


def main(argv):
    path = FLAGS.path
    df = pd.read_csv(path, sep="\t")
    x = df["# Step"]
    y = df[FLAGS.metric].rolling(window=FLAGS.window, min_periods=0).mean()
    if FLAGS.ascii:
        fig = tpl.figure()
        fig.plot(x, y)
        fig.show()
    else:
        y.plot()
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(x, y)
        fig.show()
        fig.savefig(FLAGS.out)


if __name__ == "__main__":
    app.run(main)
