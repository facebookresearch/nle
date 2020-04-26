# Copyright (c) Facebook, Inc. and its affiliates.
import concurrent.futures
import io
import unittest
import os
import struct
import sys
import time

from nle.nethack import ptyprocess


def _proc_target():
    for i in range(1000000):
        sys.stdout.write(str(i))
        sys.stdout.flush()
        time.sleep(0.1)


def read_ttyrec_header(f):
    while True:
        header = f.read(13)
        if not header:
            return
        sec, usec, length, channel = struct.unpack("<iiiB", header)
        if sec < 0 or usec < 0 or length < 1 or channel not in (0, 1):
            raise IOError("Illegal header %s" % ((sec, usec, length, channel),))
        timestamp = sec + usec * 1e-6
        yield timestamp, length, channel


class PtyProcess(unittest.TestCase):
    def test_simple(self):
        calls = 0

        def recordclosefn(filename):
            nonlocal calls
            calls += 1
            self.assertTrue(os.path.exists(filename))
            os.unlink(filename)

        p = ptyprocess.PtyProcess(_proc_target, recordclosefn=recordclosefn)
        p.fork()

        time.sleep(0.5)
        del p
        self.assertEqual(calls, 1)

    def test_recordfile(self):
        future = concurrent.futures.Future()

        def recordclosefn(filename):
            self.assertTrue(os.path.exists(filename))
            future.set_result(filename)

        p = ptyprocess.PtyProcess(_proc_target, recordclosefn=recordclosefn)
        p.fork()

        for _ in range(3):
            p.write(b"x")

        time.sleep(0.5)
        self.assertFalse(future.done())
        del p
        self.assertTrue(future.done())

        streams = [io.BytesIO(), io.BytesIO()]

        filename = future.result()
        with open(filename, "rb") as f:
            for _, length, channel in read_ttyrec_header(f):
                streams[channel].write(f.read(length))

        self.assertEqual(streams[0].getvalue().count(b"x"), 3)
        for i, c in enumerate(streams[0].getvalue().replace(b"x", b"")):
            self.assertEqual(c, ord(str(i)))
        self.assertEqual(streams[1].getvalue(), b"xxx")

        os.unlink(filename)


if __name__ == "__main__":
    unittest.main()
