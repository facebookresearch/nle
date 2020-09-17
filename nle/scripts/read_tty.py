import struct


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


if __name__ == "__main__":
    import datetime
    import sys

    def color(s, value):
        return "\033[%d;3%dm%s\033[0m" % (bool(value & 8), value & ~8, s)

    filename = sys.argv[1]
    with open(filename, "rb") as f:
        for timestamp, channel, data in ttyframes(f):
            if channel == 0:
                data = str(data)[2:-1]  # Strip b' and '
                channel = "<-"
            elif channel == 1:
                action, *_ = struct.unpack("<B", data)
                data = action
                channel = "->"

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
