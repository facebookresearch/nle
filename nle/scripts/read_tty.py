# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import atexit
import datetime
import os
import re
import signal
import struct
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "-1",
    "--no_input",
    action="store_true",
    help="Use ttyrec (not ttyrec2) format without input data",
)
parser.add_argument(
    "-c",
    "--no_unicode_csi",
    dest="unicode_csi",
    action="store_false",
    help="Don't display control sequence introducer with unicode symbol",
)
parser.add_argument("--start", default=0, type=int, help="Start at a specific frame")
parser.add_argument(
    "--end", default=float("inf"), type=int, help="Quit after a specific frame count"
)
parser.add_argument(
    "filename", default="", type=str, nargs="?", help="tty record file, or - for stdin"
)
parser.add_argument(
    "--no_pager",
    dest="use_pager",
    action="store_false",
    help="Don't pipe data to less",
)


def ttyframes(f, tty2=True):
    while True:
        if tty2:
            header = f.read(13)
        else:
            header = f.read(12)

        if not isinstance(header, bytes):
            raise IOError("File must be opened in binary mode.")

        if not header:
            return

        if tty2:
            sec, usec, length, channel = struct.unpack("<iiiB", header)
        else:
            sec, usec, length = struct.unpack("<iii", header)
            channel = 0

        if sec < 0 or usec < 0 or length < 0 or channel not in (0, 1, 2):
            raise IOError("Illegal header %s in %s" % ((sec, usec, length, channel), f))
        timestamp = sec + usec * 1e-6

        data = f.read(length)

        yield timestamp, channel, data


def getfile(filename):
    if filename == "-":
        f = os.fdopen(os.dup(0), "rb")
        os.dup2(1, 0)
        return f
    elif os.path.splitext(filename)[1] in (".bz2", ".bzip2"):
        import bz2

        return bz2.BZ2File(filename)
    elif os.path.splitext(filename)[1] in (".gz", ".gzip"):
        import gzip

        return gzip.GzipFile(filename)
    else:
        return open(filename, "rb")


def color(s, value):
    return "\033[%d;3%dm%s\033[0m" % (bool(value & 8), value & ~8, s)


CTRL_COLOR = 8  # Dark gray.
DEC_COLOR = 4  # Dark blue.
DEC_DATA_COLOR = 3  # Dark yellow.

FRAMECNT_COLOR = 2  # Dark green.
TIMESTAMP_COLOR = 7  # "Normal" color.
CHANNEL_COLOR = 2  # Dark green.
BRACES_COLOR = [11, 4, 5]  # Output: Bright yellow, input: dark blue, score: pink


# "Select Graphic Rendition" sequence.
COLOR_REGEX = re.compile(
    r"""
(?P<begin>
  (\\x1b\[)  # CSI (Control Sequence Introducer)
  (?P<bright>[0-9])
  ;3
  (?P<color>[0-9])
  m
)
(?P<data> .*?)
(?P<end> \\x1b\[0m)
""",
    re.X,
)

# Generic control sequence
CTRL_REGEX = re.compile(r"\\x1b\[ ([0-9];?) *.", re.X)

# DEC Special Graphics
DEC_REGEX = re.compile(
    r"""
(?P<begin> \\x1b\(0)
(?P<data> .*?)
(?P<end> \\x1b\(B)
""",
    re.X,
)

CSI_REGEX = re.compile(r"\\x1b\[")

CSI_UNICODE = "⌘"


def _colorsub(m):
    """Substitute color in `COLOR_REGEX`."""
    bright = int(m.group("bright"))
    value = int(m.group("color"))
    if bright:
        value |= 8
    return m.group("begin") + color(m.group("data"), value) + m.group("end")


def _ctrlsub(m):
    return color(m.group(0), CTRL_COLOR)


def _decsub(m):
    """Substitute for `DEC_REGEX`."""
    return (
        color(m.group("begin"), DEC_COLOR)
        + color(m.group("data"), DEC_DATA_COLOR)
        + color(m.group("end"), DEC_COLOR)
    )


def main():
    global FLAGS
    FLAGS = parser.parse_args()

    if not FLAGS.filename:
        parser.print_help()
        return

    if FLAGS.use_pager:
        setup_pager()

    frames = [0, 0, 0]
    with getfile(FLAGS.filename) as f:
        for timestamp, channel, data in ttyframes(f, tty2=not FLAGS.no_input):
            frames[channel] += 1

            if frames[0] < FLAGS.start:
                continue

            if frames[0] > FLAGS.end:
                return

            if channel == 0:
                arrow = "<-"
            elif channel == 1:
                char, *_ = struct.unpack("<B", data)
                data = chr(char).encode("ascii", "backslashreplace")
                arrow = "->"
            elif channel == 2:
                score, *_ = struct.unpack("<i", data)
                data = f"  {score} "
                arrow = "->"

            data = str(data)[2:-1]  # Strip b' and '

            data = re.sub(COLOR_REGEX, _colorsub, data)
            data = re.sub(CTRL_REGEX, _ctrlsub, data)
            data = re.sub(DEC_REGEX, _decsub, data)

            if FLAGS.unicode_csi:
                data = re.sub(CSI_REGEX, CSI_UNICODE, data)

            print(
                "%s %s %s%s%s%s"
                % (
                    color(str(frames), FRAMECNT_COLOR),
                    color(datetime.datetime.fromtimestamp(timestamp), TIMESTAMP_COLOR),
                    color(arrow, CHANNEL_COLOR),
                    color("{", BRACES_COLOR[channel]),
                    data,
                    color("}", BRACES_COLOR[channel]),
                )
            )


def setup_pager():
    """Pager logic. Compare pager.c in git."""
    if not os.isatty(1):
        return

    pager_env = os.environ.copy()
    pager_env["LESS"] = "FRX"
    pager = subprocess.Popen(["less"], bufsize=0, stdin=subprocess.PIPE, env=pager_env)
    os.dup2(pager.stdin.fileno(), 1)
    if os.isatty(2):
        os.dup2(pager.stdin.fileno(), 2)
    pager.stdin.close()

    def close_pager_fds():
        os.close(1)
        os.close(2)

    def wait_for_pager_atexit():
        sys.stdout.flush()
        sys.stderr.flush()
        close_pager_fds()
        pager.wait()

    signos = (
        signal.SIGINT,
        signal.SIGHUP,
        signal.SIGTERM,
        signal.SIGQUIT,
        signal.SIGPIPE,
    )

    handlers = {signo: signal.getsignal(signo) for signo in signos}

    def wait_for_pager_signal(signo, frame):
        close_pager_fds()
        pager.wait()
        for signo, handler in handlers.items():
            signal.signal(signo, handler)
        signal.raise_signal(signo)

    for signo in signos:
        signal.signal(signo, wait_for_pager_signal)
    atexit.register(wait_for_pager_atexit)


if __name__ == "__main__":
    main()
