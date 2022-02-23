import bz2
import concurrent.futures as futures
import contextlib

import numpy as np
import pytest
import torch
from test_converter import COLSROWS
from test_converter import FINALFRAME
from test_converter import FINALFRAMECOLORS
from test_converter import TIMESTAMPS
from test_converter import getfilename
from test_db import conn  # noqa: F401
from test_db import mockdata  # noqa: F401

from nle.dataset import dataset
from nle.dataset import db


class TestDataset:
    @pytest.fixture
    def db_exists(self, conn):  # noqa: F811
        """Loading this module ensure the ttyrec.db can be found by TtyrecDataset"""
        del conn

    @pytest.fixture(params=["nopool", "threadpool"])
    def pool(self, request):
        if request.param == "threadpool":
            cm = futures.ThreadPoolExecutor
        else:
            cm = contextlib.nullcontext
        with cm() as tp:
            yield tp

    def test_setup(self, conn):  # noqa: F811
        files = [db.getrow(f"{i+1}", conn=conn)[1] for i in range(9)]
        names = ["aaa", "bbb", "ccc"]
        assert files == [f"{a}/{b}.ttyrec.bz2" for a in names for b in names]

    def test_dataset_rowids(self, db_exists, pool):
        rowids = np.random.choice(9, 9, replace=False) + 1
        data = dataset.TtyrecDataset(batch_size=9, threadpool=pool, rowids=rowids)
        mb = next(iter(data))
        files = mb["rowids"]
        assert len(mb) == 6

        np.testing.assert_array_equal(np.unique(files.numpy()[:, 0]), np.arange(1, 10))

    def test_minibatches(self, db_exists, pool):
        data = dataset.TtyrecDataset(
            seq_length=50,
            batch_size=4,
            threadpool=pool,
            rowids=range(1, 10),
            shuffle=False,
        )
        # starting rowids = [TTYREC, TTYREC, TTYREC, TTYREC2]

        mb = next(iter(data))
        for name, array in mb.items():
            if name in ("rowids",):
                continue
            # Test first three rows are the same, and differ from from fourth
            np.testing.assert_array_equal(array[0], array[1])
            np.testing.assert_array_equal(array[0], array[2])
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal, array[0], array[3]
            )

        # Check reseting occured
        reset = np.where(mb["done"][3] == 1)[0][0]
        assert reset == 31

        # Check the data at location is the same. Note reset occurs for batch 4
        seq = 10
        for name, array in mb.items():
            if name in ("done", "rowids"):
                continue
            np.testing.assert_array_equal(array[3][:seq], array[3][reset : reset + seq])

        # No leading 1s
        assert (mb["done"][:, 0] == 0).all()

    def test_get_ttyrec(self, db_exists, pool):
        data = dataset.TtyrecDataset(
            seq_length=100, batch_size=1, rowids=[4, 5], threadpool=pool
        )

        mb = next(iter(data))
        reset = np.where(mb["done"][0] == 1)[0][0]
        assert reset == 31

        for chunk_size in [5, 50]:
            chunks = data.get_ttyrec(4, chunk_size=chunk_size)

            groups = {k: [] for k in mb.keys()}
            for c in chunks:
                for k in c:
                    groups[k].append(c[k])
            concat_chunks = {k: torch.cat(t, 1) for k, t in groups.items()}
            for k in mb.keys():
                c = concat_chunks[k]
                m = mb[k][0]
                np.testing.assert_array_equal(c.numpy()[0, :reset], m.numpy()[:reset])
                np.testing.assert_equal(c.numpy()[0, reset:], 0)

    def test_char_frame(self, db_exists, pool):
        # rowids [1,2,3] are all the same tty_rec, rowid 4 is different
        rowids = [1, 2, 3, 4, 1, 2, 3, 4]
        seq_length = 20
        data = dataset.TtyrecDataset(
            seq_length=seq_length, rows=25, batch_size=3, rowids=rowids, threadpool=pool
        )

        with open(getfilename(FINALFRAME), "r") as f:
            lines = [line.rstrip() for line in f.readlines()]

        with open(getfilename(FINALFRAMECOLORS), "r") as f:
            colorlines = [line.rstrip().split(",") for line in f.readlines()]

        with open(getfilename(COLSROWS)) as f:
            colsrows = [tuple(int(i) for i in line.split()) for line in f]

        with bz2.BZ2File(getfilename(TIMESTAMPS)) as f:
            saved_timestamps = [float(line) * 1e6 for line in f]

        i = 0
        for mb in data:
            chars, colors, curs, ts, done, _ = mb.values()
            batch_ids, seq_ids = np.where(done.numpy() == 1)
            seq = seq_length if not seq_ids.any() else seq_ids[0]
            for j in range(seq):
                np.testing.assert_array_equal(colsrows[i][0], curs[:, j, 1])
                np.testing.assert_array_equal(colsrows[i][1], curs[:, j, 0])
                np.testing.assert_array_almost_equal(
                    saved_timestamps[i], ts[:, j], decimal=0
                )

                i += 1
            if seq_ids.any():
                np.testing.assert_array_equal(batch_ids, np.arange(3))
                np.testing.assert_array_equal(seq_ids, seq_ids[0])

                final_frame = chars[0][seq_ids[0] - 1].numpy()
                char_frame = [
                    row.tobytes().decode("utf-8").rstrip() for row in final_frame
                ]
                color_frame = colors[0][seq_ids[0] - 1].numpy()

                for actual, expected in zip(char_frame, lines):
                    assert actual == expected
                for actual, expected in zip(color_frame, colorlines):
                    for col_a, col_e in zip(actual, expected):
                        assert col_a == int(col_e)
                break
            if i > 2431:
                raise AssertionError  # this should have ended!

    def test_start_and_end(self, db_exists):
        # rowids [1,2,3] are all the same tty_rec, rowid 4 is different
        data_1 = dataset.TtyrecDataset(seq_length=100, batch_size=3, rowids=range(1, 4))
        data_2 = dataset.TtyrecDataset(seq_length=100, batch_size=1, rowids=[4, 4, 4])

        mb1 = next(iter(data_1))
        mb2 = next(iter(data_2))

        for k in mb1:
            if k == "rowids":
                continue
            array1 = mb1[k]
            array2 = mb2[k]
            # Test first three rows are the same, and differ from from fourth
            np.testing.assert_array_equal(array1[0], array1[1])
            np.testing.assert_array_equal(array1[0], array1[2])
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal, array1[0], array2[0]
            )

    @pytest.mark.parametrize("batch_size", [2, 5, 9])
    def test_no_looping(self, db_exists, pool, batch_size):
        # We expect as many done as datapoint - batchsize) Expected behaviour:
        data = dataset.TtyrecDataset(
            seq_length=100,
            batch_size=batch_size,
            threadpool=pool,
            rowids=range(1, 10),
        )
        done = 0
        for mb in data:
            done += np.sum(mb["done"].numpy() == 1)
            if done > 9 - batch_size:
                break
        assert done == 9 - batch_size

    @pytest.mark.parametrize("batch_size", [1, 5, 9])
    def test_shuffle(self, db_exists, batch_size):
        data = dataset.TtyrecDataset(
            seq_length=1000,
            batch_size=batch_size,
            threadpool=None,
            rowids=range(1, 10),
        )

        def get_data():
            """Get all the data from minibatches and concat into full data batches"""
            mbs = []
            for mb in data:
                mbs.append([t.clone().detach() for t in mb.values()])
            return [torch.cat(tensor_list, dim=1) for tensor_list in zip(*mbs)]

        b1 = get_data()
        b2 = get_data()
        for a1, a2 in zip(b1, b2):
            np.testing.assert_array_equal(a1, a2)

        data.shuffle = True
        b3 = get_data()
        b4 = get_data()
        # Note: the last tensor in the batch is the rowid
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, b3[-1], b2[-1]
        )
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, b3[-1], b4[-1]
        )

        data.shuffle = False
        b5 = get_data()
        for a1, a5 in zip(b1, b5):
            np.testing.assert_array_equal(a1, a5)

    def test_sql(self, db_exists, pool):
        sql = "SELECT rowid, path FROM ttyrecs WHERE rowid >= 6 ORDER BY rowid"
        data = dataset.TtyrecDataset(
            seq_length=100, batch_size=2, threadpool=pool, sql_subset=sql
        )

        data2 = dataset.TtyrecDataset(
            seq_length=100, batch_size=2, threadpool=pool, rowids=[6, 7, 8, 9]
        )

        for _, (mb1, mb2) in enumerate(zip(data, data2)):
            for k in mb1.keys():
                np.testing.assert_array_equal(mb1[k], mb2[k])

    def test_multipart_game(self, db_exists, pool):
        # sql1 -> select by game: all of user "ccc" ttyrecs (7, 8, 9) as one game
        # eg: [(1, /path/to/A), (1, /path/to/B), (1, /path/to/C)]
        sql1 = (
            "SELECT games.gameid, ttyrecs.path FROM ttyrecs "
            "INNER JOIN games ON ttyrecs.gameid=games.gameid "
        )
        data1 = dataset.TtyrecDataset(
            seq_length=100, batch_size=1, threadpool=pool, sql_subset=sql1
        )

        # sql2 -> select by ttyrec: all of user "ccc" ttyrecs (7, 8, 9)
        # eg: [(7, /path/to/A), (8, /path/to/B), (9, /path/to/C)]
        sql2 = 'SELECT rowid, path FROM ttyrecs WHERE user="ccc" ORDER BY rowid'
        data2 = dataset.TtyrecDataset(
            seq_length=100, batch_size=1, threadpool=pool, sql_subset=sql2
        )

        for _, (mb1, mb2) in enumerate(zip(data1, data2)):
            for k in mb1.keys():
                if k == "rowids":
                    assert np.all(np.isin(mb1[k], [0, 1]))
                elif k == "done":
                    np.testing.assert_equal(mb1[k], np.zeros_like(mb1[k]))
                else:
                    np.testing.assert_array_equal(mb1[k], mb2[k])

    def test_dataset_metadata(self, db_exists, pool):
        # Test we can retrieve metadata from database and access it later
        # Everything after the gameid, path should be stored as a list in metadata
        sql1 = (
            "SELECT games.gameid, ttyrecs.path, games.death, games.points, "
            "ttyrecs.rowid FROM ttyrecs INNER JOIN games "
            "ON ttyrecs.gameid=games.gameid "
        )
        data1 = dataset.TtyrecDataset(
            seq_length=100, batch_size=1, threadpool=pool, sql_subset=sql1
        )
        rowids = next(iter(data1))["rowids"]
        for i, ttyrec in enumerate([7, 8, 9]):
            rowid = rowids.numpy()[0][0]
            assert data1.get_meta(rowid)[i] == ("ascended", 999, ttyrec)
