# Copyright (c) Facebook, Inc. and its affiliates.
import struct
import os
import re


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

        if sec < 0 or usec < 0 or length < 0 or channel not in (0, 1):
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


if __name__ == "__main__":
    import datetime
    import sys

    def color(s, value):
        return "\033[%d;3%dm%s\033[0m" % (bool(value & 8), value & ~8, s)

    filename = sys.argv[1]
    with getfile(filename) as f:
        for timestamp, channel, data in ttyframes(f):
            if channel == 0:
                data = str(data)[2:-1]  # Strip b' and '
                channel = "<-"
            elif channel == 1:
                action, *_ = struct.unpack("<B", data)
                data = action
                channel = "->"

            data = re.sub(r"\\x1b\[([0-9];?)*.", lambda m: color(m.group(0), 8), data)
            data = re.sub(
                r"(\\x1b\(0)(.*?)(\\x1b\(B)",
                lambda m: (
                    color(m.group(1), 4) + color(m.group(2), 3) + color(m.group(3), 4)
                ),
                data,
            )

            print(
                "%s %s%s%s%s"
                % (
                    datetime.datetime.fromtimestamp(timestamp),
                    color(channel, 2),
                    color("{", 11),
                    data,
                    color("}", 11),
                )
            )
