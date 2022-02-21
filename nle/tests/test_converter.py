import bz2
import os
import re

import numpy as np
import pytest
from memory_profiler import memory_usage

from nle.dataset import Converter

# From
#   https://alt.org/nethack/trd/?file=https://s3.amazonaws.com/altorg/ttyrec/Anarchos/2020-10-03.17:27:10.ttyrec.bz2  # noqa: B950
TTYREC_2020 = "2020-10-03.17_27_10.ttyrec.bz2"
COLSROWS = "2020-10-03.17_27_10.rowscols.txt"
TIMESTAMPS = "2020-10-03.17_27_10.timestamps.txt.bz2"
FINALFRAME = "2020-10-03.17_27_10.finalframe.txt"
FINALFRAMECOLORS = "2020-10-03.17_27_10.finalframe.colors.txt"

# From
#  https://alt.org/nethack/trd/?file=https://s3.amazonaws.com/altorg/ttyrec/Qvazzler/2009-02-05.10:33:51.ttyrec.bz2  # noqa: B950
# This ttyrec uses IBMGraphics (code page 437)
TTYREC_IBMGRAPHICS = "2009-02-05.10_33_51.ttyrec.bz2"
TTYREC_IBMGRAPHICS_FRAME_10 = "2009-02-05.10_33_51.frame.10.txt"

# From
#  https://alt.org/nethack/trd/?file=https://s3.amazonaws.com/altorg/ttyrec/moo22/2018-09-27.00:20:39.ttyrec.bz2  # noqa: B950
TTYREC_2018 = "2018-09-27.00_20_39.ttyrec.bz2"

# From
#  https://alt.org/nethack/trd/?file=https://s3.amazonaws.com/altorg/ttyrec/CodeMagic/2020-10-16.00:11:28.ttyrec.bz2  # noqa: B950
# This ttyrec uses DECGraphics (https://en.wikipedia.org/wiki/DEC_Special_Graphics)
TTYREC_DECGRAPHICS = "2020-10-16.00_11_28.ttyrec.bz2"

SEQ_LENGTH = 20
ROWS = 25
COLUMNS = 80


def getfilename(filename):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)


def load_and_convert(converter, ttyrec, chars, colors, cursors, timestamps, actions):
    converter.load_ttyrec(ttyrec)
    remaining = converter.convert(chars, colors, cursors, timestamps, actions)
    while remaining == 0:
        remaining = converter.convert(chars, colors, cursors, timestamps, actions)


class TestConverter:
    def test_is_loaded(self):
        converter = Converter(ROWS, COLUMNS)
        assert not converter.is_loaded()
        converter.load_ttyrec(getfilename(TTYREC_2020))
        assert converter.is_loaded()

    def test_no_memleak(self):
        chars = np.zeros((SEQ_LENGTH, ROWS, COLUMNS), dtype=np.uint8)
        colors = np.zeros((SEQ_LENGTH, ROWS, COLUMNS), dtype=np.int8)
        cursors = np.zeros((SEQ_LENGTH, 2), dtype=np.uint16)
        timestamps = np.zeros((SEQ_LENGTH,), dtype=np.int64)
        actions = np.zeros((SEQ_LENGTH), dtype=np.uint8)
        ttyrec = getfilename(TTYREC_2020)
        converter = Converter(ROWS, COLUMNS)

        def convert_n_times(n):
            for _ in range(n):
                load_and_convert(
                    converter, ttyrec, chars, colors, cursors, timestamps, actions
                )

        memory_list = memory_usage((convert_n_times, (100,), {}))
        # After warmup the last few iterations should be constant memory
        memory_array = np.array(memory_list[3:])
        memory_change = (memory_array / memory_array[0]) - 1
        assert max(memory_change) < 0.001  # 0.1 per cent

    def test_ttyrec_with_extra_data(self, seq_length=500):
        converter = Converter(ROWS, COLUMNS)

        chars = np.zeros((seq_length, ROWS, COLUMNS), dtype=np.uint8)
        colors = np.zeros((seq_length, ROWS, COLUMNS), dtype=np.int8)
        cursors = np.zeros((seq_length, 2), dtype=np.uint16)
        timestamps = np.zeros((seq_length,), dtype=np.int64)
        actions = np.zeros((seq_length), dtype=np.uint8)

        converter.load_ttyrec(getfilename(TTYREC_2018))
        remaining = converter.convert(chars, colors, cursors, timestamps, actions)
        assert remaining == 165

    def test_data(self):
        converter = Converter(ROWS, COLUMNS)
        assert converter.rows == ROWS
        assert converter.cols == COLUMNS

        chars = np.zeros((SEQ_LENGTH, ROWS, COLUMNS), dtype=np.uint8)
        colors = np.zeros((SEQ_LENGTH, ROWS, COLUMNS), dtype=np.int8)
        cursors = np.zeros((SEQ_LENGTH, 2), dtype=np.uint16)
        timestamps = np.zeros((SEQ_LENGTH,), dtype=np.int64)
        actions = np.zeros((SEQ_LENGTH), dtype=np.uint8)

        converter.load_ttyrec(getfilename(TTYREC_2020))
        with open(getfilename(COLSROWS)) as f:
            colsrows = [tuple(int(i) for i in line.split()) for line in f]

        with bz2.BZ2File(getfilename(TIMESTAMPS)) as f:
            saved_timestamps = [float(line) for line in f]

        num_frames = 0
        while True:
            remaining = converter.convert(chars, colors, cursors, timestamps, actions)
            for (row, col), ts in zip(
                cursors[: SEQ_LENGTH - remaining], timestamps[: SEQ_LENGTH - remaining]
            ):
                assert (col, row) == colsrows[num_frames]
                assert pytest.approx(float(ts) / 1e6) == saved_timestamps[num_frames]
                assert 0 <= col < converter.cols
                assert 0 <= row < converter.rows
                num_frames += 1
            if remaining > 0:
                break

        assert num_frames == len(colsrows)
        final_index = SEQ_LENGTH - remaining - 1
        with open(getfilename(FINALFRAME)) as f:
            for row, line in enumerate(f):

                actual = chars[final_index][row].tobytes().decode("utf-8").rstrip()
                assert actual == line.rstrip()
        with open(getfilename(FINALFRAMECOLORS)) as f:
            for row, line in enumerate(f):
                actual = ",".join(str(c) for c in colors[final_index][row])
                assert actual == line.rstrip()

    def test_noexist(self):
        fn = "/does/not/exist.txt"
        converter = Converter(25, 80)
        with pytest.raises(
            FileNotFoundError, match=r"\[Errno 2\] No such file or directory: '%s'" % fn
        ):
            converter.load_ttyrec(fn)

    def test_illegal_buffers(self):
        converter = Converter(ROWS, COLUMNS)
        converter.load_ttyrec(getfilename(TTYREC_2020))

        chars = np.zeros((10, ROWS, COLUMNS), dtype=np.uint8)
        colors = np.zeros((10, ROWS, COLUMNS), dtype=np.int8)
        cursors = np.zeros((11, 2), dtype=np.uint16)
        actions = np.zeros((10), dtype=np.uint8)
        timestamps = np.zeros((10,), dtype=np.int64)
        with pytest.raises(
            ValueError,
            match=re.escape("Array has wrong shape (expected [ 10 2 ], got [ 11 2 ])"),
        ):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.zeros((10, ROWS, COLUMNS), dtype=np.uint8)
        colors = np.zeros((10, ROWS, COLUMNS - 1), dtype=np.int8)
        cursors = np.zeros((10, 2), dtype=np.uint16)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Array has wrong shape (expected [ 10 25 80 ], got [ 10 25 79 ])"
            ),
        ):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.zeros((10, ROWS - 1, COLUMNS), dtype=np.uint8)
        colors = np.zeros((10, ROWS - 1, COLUMNS), dtype=np.int8)
        cursors = np.zeros((10, 2), dtype=np.uint16)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Array has wrong shape (expected [ 10 25 80 ], got [ 10 24 80 ])"
            ),
        ):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.zeros((11, ROWS, COLUMNS), dtype=np.uint8)
        colors = np.zeros((11, ROWS, COLUMNS), dtype=np.int8)
        cursors = np.zeros((11, 2), dtype=np.uint16)
        actions = np.zeros((10), dtype=np.uint8)
        timestamps = np.zeros((10,), dtype=np.int64)
        with pytest.raises(
            ValueError,
            match=re.escape("Array has wrong shape (expected [ 11 ], got [ 10 ])"),
        ):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.zeros((10, ROWS, COLUMNS), dtype=np.uint8)
        colors = np.zeros((10, ROWS, COLUMNS), dtype=np.int8)
        cursors = np.zeros((10, 3), dtype=np.uint16)
        with pytest.raises(
            ValueError,
            match=re.escape("Array has wrong shape (expected [ 10 2 ], got [ 10 3 ])"),
        ):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.zeros((10, ROWS, COLUMNS, 7), dtype=np.uint8)
        cursors = np.zeros((10, 2), dtype=np.uint16)
        with pytest.raises(
            ValueError,
            match=r"Array has wrong number of dimensions \(expected 3, got 4\)",
        ):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.zeros((10, ROWS, COLUMNS), dtype=np.uint8)
        cursors = np.zeros((10, 2, 1), dtype=np.uint16)
        with pytest.raises(
            ValueError,
            match=r"Array has wrong number of dimensions \(expected 2, got 3\)",
        ):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.zeros((10, ROWS, COLUMNS), dtype=np.float32)
        cursors = np.zeros((10, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"Buffer dtype mismatch"):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.zeros((10, ROWS, COLUMNS), dtype=np.uint8)
        cursors = np.zeros((10, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"Buffer dtype mismatch"):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = "Hello"
        cursors = np.zeros((10, 2), dtype=np.uint16)
        with pytest.raises(ValueError, match=r"Numpy array required"):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.uint(8)
        with pytest.raises(ValueError, match=r"Numpy array required"):
            converter.convert(chars, colors, cursors, timestamps, actions)

        chars = np.zeros((10, ROWS, COLUMNS), dtype=np.uint8)
        timestamps = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        with pytest.raises(ValueError, match=r"Numpy array required"):
            converter.convert(chars, colors, cursors, timestamps, actions)

    def _test_ibm_graphics(self):
        seq_length = 10
        converter = Converter(ROWS, COLUMNS)

        chars = np.zeros((seq_length, ROWS, COLUMNS), dtype=np.uint8)
        colors = np.zeros((seq_length, ROWS, COLUMNS), dtype=np.int8)
        cursors = np.zeros((seq_length, 2), dtype=np.uint16)
        actions = np.zeros((seq_length), dtype=np.uint8)
        timestamps = np.zeros((seq_length,), dtype=np.int64)

        converter.load_ttyrec(getfilename(TTYREC_IBMGRAPHICS))
        assert converter.convert(chars, colors, cursors, timestamps, actions) == 0

        with open(getfilename(TTYREC_IBMGRAPHICS_FRAME_10)) as f:
            for row, line in enumerate(f):
                actual = chars[-1][row].tobytes().decode("utf-8").rstrip()
                assert actual == line.rstrip()
