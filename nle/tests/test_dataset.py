import bz2
import concurrent.futures as futures
import contextlib

import numpy as np
import pytest
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
            try:
                cm = contextlib.nullcontext
            except AttributeError:
                # `nullcontext`` is Python 3.7+ feature.
                @contextlib.contextmanager
                def nullcontext(*args, **kwargs):
                    yield

                cm = lambda: nullcontext()
        with cm() as tp:
            yield tp

    def test_setup(self, conn):  # noqa: F811
        files = [db.get_row(f"{i+1}", conn=conn)[1] for i in range(9)]
        names = ["aaa", "bbb", "ccc"]
        assert files == [f"{a}/{b}.ttyrec.bz2" for a in names for b in names]

    def test_dataset_gameids(self, db_exists, pool):
        gameids = np.random.choice(7, 7, replace=False) + 1
        data = dataset.TtyrecDataset(
            "basictest", batch_size=7, threadpool=pool, gameids=gameids
        )
        mb = next(iter(data))
        files = mb["gameids"]
        assert len(mb) == 6

        np.testing.assert_array_equal(np.unique(files[:, 0]), np.arange(1, 8))

    def test_minibatches(self, db_exists, pool):
        data = dataset.TtyrecDataset(
            "basictest",
            seq_length=50,
            batch_size=4,
            threadpool=pool,
            gameids=range(1, 8),
            shuffle=False,
        )
        # starting gameids = [TTYREC, TTYREC, TTYREC, TTYREC2]

        mb = next(iter(data))
        for name, array in mb.items():
            if name in ("gameids",):
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
            if name in ("done", "gameids"):
                continue
            np.testing.assert_array_equal(array[3][:seq], array[3][reset : reset + seq])

        # No leading 1s
        assert (mb["done"][:, 0] == 0).all()

    def test_get_ttyrec(self, db_exists, pool):
        data = dataset.TtyrecDataset(
            "basictest",
            seq_length=100,
            batch_size=1,
            gameids=[4, 5],
            threadpool=pool,
            shuffle=False,
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
            concat_chunks = {k: np.concatenate(t, axis=1) for k, t in groups.items()}
            for k in mb.keys():
                c = concat_chunks[k]
                m = mb[k][0]
                np.testing.assert_array_equal(c[0, :reset], m[:reset])
                np.testing.assert_equal(c[0, reset:], 0)

    def test_char_frame(self, db_exists, pool):
        # gameids [1,2,3] are all the same tty_rec, rowid 4 is different
        gameids = [1, 2, 3, 4, 1, 2, 3, 4]
        seq_length = 20
        data = dataset.TtyrecDataset(
            "basictest",
            seq_length=seq_length,
            rows=25,
            batch_size=3,
            gameids=gameids,
            threadpool=pool,
            shuffle=False,
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
            batch_ids, seq_ids = np.where(done == 1)
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

                final_frame = chars[0][seq_ids[0] - 1]
                char_frame = [
                    row.tobytes().decode("utf-8").rstrip() for row in final_frame
                ]
                color_frame = colors[0][seq_ids[0] - 1]

                for actual, expected in zip(char_frame, lines):
                    assert actual == expected
                for actual, expected in zip(color_frame, colorlines):
                    for col_a, col_e in zip(actual, expected):
                        assert col_a == int(col_e)
                break
            if i > 2431:
                raise AssertionError  # this should have ended!

    def test_start_and_end(self, db_exists):
        # gameids [1,2,3] are all the same tty_rec, rowid 4 is different
        data_1 = dataset.TtyrecDataset(
            "basictest",
            seq_length=100,
            batch_size=3,
            gameids=range(1, 4),
            shuffle=False,
        )
        data_2 = dataset.TtyrecDataset(
            "basictest", seq_length=100, batch_size=1, gameids=[4, 4, 4], shuffle=False
        )

        mb1 = next(iter(data_1))
        mb2 = next(iter(data_2))

        for k in mb1:
            if k == "gameids":
                continue
            array1 = mb1[k]
            array2 = mb2[k]
            # Test first three rows are the same, and differ from from fourth
            np.testing.assert_array_equal(array1[0], array1[1])
            np.testing.assert_array_equal(array1[0], array1[2])
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal, array1[0], array2[0]
            )

    @pytest.mark.parametrize("batch_size", [2, 5, 7])
    def test_no_looping(self, db_exists, pool, batch_size):
        # We expect as many done as (len(gameids) - batchsize)
        data = dataset.TtyrecDataset(
            "basictest",
            seq_length=100,
            batch_size=batch_size,
            threadpool=pool,
            gameids=range(1, 8),
        )
        done = 0
        for mb in data:
            done += np.sum(mb["done"] == 1)
            if done > 7 - batch_size:
                break
        assert done == 7 - batch_size

    @pytest.mark.parametrize("batch_size", [1, 3, 6])
    def test_shuffle(self, db_exists, batch_size):
        data = dataset.TtyrecDataset(
            "basictest",
            seq_length=1000,
            batch_size=batch_size,
            threadpool=None,
            gameids=(1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1),
            shuffle=False,
        )

        def get_data():
            """Get all the data from minibatches and concat into full data batches"""
            mbs = []
            gameids = []
            for mb in data:
                mbs.append([t.copy() for t in mb.values()])
                gameids.append(mb["gameids"].copy())
            return [
                np.concatenate(array_list, axis=1) for array_list in zip(*mbs)
            ], np.concatenate(gameids)

        b1, g1 = get_data()
        b2, g2 = get_data()
        for a1, a2 in zip(b1, b2):
            np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(g1, g2)

        data.shuffle = True
        b3, g3 = get_data()
        b4, g4 = get_data()
        # Note: the last tensor in the batch is the gameid
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, g3, g2)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, g3, g4)

        data.shuffle = False
        b5, g5 = get_data()
        for a1, a5 in zip(b1, b5):
            np.testing.assert_array_equal(a1, a5)
        np.testing.assert_array_equal(g1, g5)

    def test_sql(self, db_exists, pool):

        sql = """
            SELECT gameid FROM datasets
            WHERE gameid >=6 AND dataset_name='basictest'
            """
        data = dataset.TtyrecDataset(
            "basictest",
            seq_length=100,
            batch_size=2,
            threadpool=pool,
            shuffle=False,
            subselect_sql=sql,
        )

        data2 = dataset.TtyrecDataset(
            "basictest",
            seq_length=100,
            batch_size=2,
            threadpool=pool,
            gameids=[6, 7],
            shuffle=False,
        )
        data.populate_metadata()
        data2.populate_metadata()
        assert len(data.get_meta(6)) == 27
        assert data.get_meta(6) == data2.get_meta(6)

        for _, (mb1, mb2) in enumerate(zip(data, data2)):
            for k in mb1.keys():
                np.testing.assert_array_equal(mb1[k], mb2[k])

    def test_multipart_game(self, db_exists, pool):
        # This test selects a multipart game

        # Use sql select all of user "ccc" ttyrecs (3 ttyrecs, gameid 7) as one game
        # eg: [(7, 0, /path/to/A), (7, 1, /path/to/B), (7, 2, /path/to/C)]
        sql1 = "SELECT gameid FROM games WHERE games.name='ccc'"
        data1 = dataset.TtyrecDataset(
            "basictest",
            seq_length=100,
            batch_size=1,
            threadpool=pool,
            subselect_sql=sql1,
        )

        # Hack data2 to treat of user "ccc" ttyrecs (7, 8, 9) as separate games
        # eg: [(0, 7, /path/to/A), (1, 7, /path/to/B), (2, 7, /path/to/C)]
        data2 = dataset.TtyrecDataset(
            "basictest",
            seq_length=100,
            batch_size=1,
            threadpool=pool,
            subselect_sql=sql1,
        )
        ccc_gameid = data2._gameids[0]
        data2._games[1000] = [data2._games[ccc_gameid][0]]
        data2._games[2000] = [data2._games[ccc_gameid][1]]
        data2._games[3000] = [data2._games[ccc_gameid][2]]
        data2._gameids = [1000, 2000, 3000]

        for _, (mb1, mb2) in enumerate(zip(data1, data2)):
            for k in mb1.keys():
                if k == "gameids":
                    assert np.all(np.isin(mb1[k], [0, 7]))
                elif k == "done":
                    np.testing.assert_equal(mb1[k], np.zeros_like(mb1[k]))
                else:
                    np.testing.assert_array_equal(mb1[k], mb2[k])

    def test_dataset_metadata(self, db_exists, pool):
        # Test we can retrieve metadata from database and access it later
        # Everything after the gameid, path should be stored as a list in metadata
        sql1 = """
            SELECT games.gameid FROM games
            INNER JOIN datasets
            ON datasets.gameid = games.gameid
            WHERE datasets.dataset_name='basictest'
            AND games.name="ccc"
            """
        data1 = dataset.TtyrecDataset(
            "basictest",
            seq_length=100,
            batch_size=1,
            threadpool=pool,
            subselect_sql=sql1,
        )
        data1.populate_metadata()
        gameids = next(iter(data1))["gameids"]
        for _ in [7, 8, 9]:
            rowid = gameids[0][0]
            # These are SQLiteRows, which can be accessed
            # lists or dictionaries.
            assert data1.get_meta(rowid)[17] == "ascended"
            assert data1.get_meta(rowid)["death"] == "ascended"
            assert data1.get_meta(rowid)[2] == 999
            assert data1.get_meta(rowid)["points"] == 999
