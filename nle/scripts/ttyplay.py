#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import bz2
import gzip
import os
import re
import select
import struct
import termios
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "-1",
    "--no_input",
    action="store_true",
    help="Use ttyrec (not ttyrec2) format without input data",
)
parser.add_argument(
    "-n",
    "--no_wait",
    action="store_true",
    help="Ignore timestamp data and don't wait between frames",
)
parser.add_argument(
    "-s", "--speed", default=1.0, type=float, help="Set playback speed multiplier"
)
parser.add_argument(
    "-p", "--peek", action="store_true", help="Peek mode (like tail -f)"
)
parser.add_argument(
    "filename", default="-", type=str, nargs="?", help="tty record file, or - for stdin"
)
parser.add_argument("--start", default=0, type=int, help="Start at a specific frame")
parser.add_argument(
    "--end", default=float("inf"), type=int, help="Quit after a specific frame count"
)


def wait(diff, speed, drift=0.0):
    jump = 0

    if FLAGS.no_wait:
        return speed, drift, jump

    start = time.time()

    diff /= speed
    diff = max(diff - drift, 0.0)

    rlist, _, _ = select.select((0,), (), (), diff)
    if rlist:
        c = os.read(0, 1)
        if c in (b"+", b"f"):
            speed *= 2
        elif c in (b"-", b"s"):
            speed /= 2
        elif c == b"1":
            speed = 1
        elif c == b"h":
            jump = -1
        elif c == b"l":
            jump = 1
        elif c == b" ":
            select.select((0,), (), ())
        drift = 0.0
    else:
        stop = time.time()
        if diff == 0.0:
            diff = -drift
        drift = (stop - start) - diff
    return speed, drift, jump


def read_header(f, peek=False, no_input=False):
    while True:
        if no_input:
            header = f.read(12)
        else:
            header = f.read(13)

        if not header:
            if peek:
                # Never return, just wait for more data.
                time.sleep(0.25)
                continue
            return

        if no_input:
            sec, usec, length = struct.unpack("<iii", header)
            channel = 0
        else:
            sec, usec, length, channel = struct.unpack("<iiiB", header)

        if sec < 0 or usec < 0 or length < 1 or channel not in (0, 1):
            raise IOError("Illegal header %s" % ((sec, usec, length, channel),))
        timestamp = sec + usec * 1e-6
        yield timestamp, length, channel


CLRCODE = re.compile(rb"\033\[2?J")  # https://stackoverflow.com/a/37778152/1136208


def process(f):
    speed = FLAGS.speed
    drift = 0.0
    prev = None
    jump = 0

    # Store the header positions before clear screen commands,
    # the timestamp before.
    clrscreen = []

    lastpos = 0
    frame = 0

    for timestamp, length, channel in read_header(
        f, peek=FLAGS.peek, no_input=FLAGS.no_input
    ):
        data = f.read(length)
        frame += 1

        if frame < FLAGS.start:
            continue

        if channel == 1:  # Input channel.
            continue

        if jump == 0 and prev is not None:
            speed, drift, jump = wait(timestamp - prev, speed, drift)

        if CLRCODE.search(data):
            clrscreen.append((lastpos, prev))
            if jump > 0:
                jump = 0

        lastpos = f.seek(0, os.SEEK_CUR)

        prev = timestamp

        if jump > 0:
            continue
        if jump < 0:
            jump = 0

            if not clrscreen:
                clrscreen.append((0, None))

            while prev is not None and timestamp - prev <= 0.005:
                lastpos, prev = clrscreen.pop()
            if lastpos:
                prev = timestamp

            f.seek(lastpos, os.SEEK_SET)
            continue

        os.write(1, data)

        if frame > FLAGS.end:
            return


def main():
    global FLAGS
    FLAGS = parser.parse_args()

    if FLAGS.filename == "-":
        f = os.fdopen(os.dup(0), "rb")
        os.dup2(1, 0)
    elif os.path.splitext(FLAGS.filename)[1] in (".bz2", ".bzip2"):
        f = bz2.BZ2File(FLAGS.filename)
    elif os.path.splitext(FLAGS.filename)[1] in (".gz", ".gzip"):
        f = gzip.GzipFile(FLAGS.filename)
    else:
        f = open(FLAGS.filename, "rb")

    old = termios.tcgetattr(0)
    new = termios.tcgetattr(0)

    # Set to unbuffered, no echo.
    new[3] &= ~(termios.ICANON | termios.ECHO | termios.ECHONL)  # lflags
    termios.tcsetattr(0, termios.TCSANOW, new)

    try:
        if FLAGS.peek:
            # Skip all previous data.
            for _, length, _ in read_header(f, peek=False):
                f.seek(length, os.SEEK_CUR)
            FLAGS.no_wait = True
        process(f)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(0, termios.TCSANOW, old)
        f.close()


if __name__ == "__main__":
    main()
