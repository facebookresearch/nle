import os
import sqlite3
import threading
from collections import defaultdict
from functools import partial

import numpy as np

from nle import _pyconverter as converter
from nle import dataset as nld


def convert_frames(
    converter,
    chars,
    colors,
    curs,
    timestamps,
    actions,
    scores,
    resets,
    gameids,
    load_fn,
):
    """Convert frames for a single batch entry.

    :param converter: A ttychars.Converter object with a loaded file.
    :param chars: Array of characters -  np.array(np.uint8) [ SEQ x ROW x COL]
    :param colors: Array of colors -  np.array(np.uint8) [ SEQ x ROW x COL]
    :param curs: Array of cursors -  np.array(np.int16) [ SEQ x 2 ]
    :param timestamps: Array of timestamps -  np.array(np.int64) [ SEQ ]
    :param actions: Array of actions at t in response to output at t
        - np.array(np.uint8) [ SEQ ]
    :param scores: Array of in-game scores -  np.array(np.int32) [ SEQ ]
    :param resets: Array of resets -  np.array(np.uint8) [ SEQ ]
    :param gameids: Array of the gameid of each frame - np.array(np.int32) [ SEQ ]
    :param load_fn: A callback that loads the next file into a converter:
        sig: load_fn(converter) -> bool is_success

    """

    resets[0] = 0
    while True:
        remaining = converter.convert(chars, colors, curs, timestamps, actions, scores)
        end = np.shape(chars)[0] - remaining

        resets[1:end] = 0
        gameids[:end] = converter.gameid
        if remaining == 0:
            return

        # There still space in the buffers; load a new ttyrec and carry on.
        chars = chars[-remaining:]
        colors = colors[-remaining:]
        curs = curs[-remaining:]
        timestamps = timestamps[-remaining:]
        actions = actions[-remaining:]
        scores = scores[-remaining:]
        resets = resets[-remaining:]
        gameids = gameids[-remaining:]
        if load_fn(converter):
            if converter.part == 0:
                resets[0] = 1
        else:
            chars.fill(0)
            colors.fill(0)
            curs.fill(0)
            timestamps.fill(0)
            actions.fill(0)
            scores.fill(0)
            resets.fill(0)
            gameids.fill(0)
            return


def _ttyrec_generator(
    batch_size, seq_length, rows, cols, load_fn, map_fn, ttyrec_version
):
    """A generator to fill minibatches with ttyrecs.

    :param load_fn: a function to load the next ttyrec into a converter.
       load_fn(ttyrecs.Converter conv) -> bool is_success
    :param map_fn: a function that maps a series of iterables through a fn.
       map_fn(fn, *iterables) -> <generator> (can use built-in map)

    """
    chars = np.zeros((batch_size, seq_length, rows, cols), dtype=np.uint8)
    colors = np.zeros((batch_size, seq_length, rows, cols), dtype=np.int8)
    cursors = np.zeros((batch_size, seq_length, 2), dtype=np.int16)
    timestamps = np.zeros((batch_size, seq_length), dtype=np.int64)
    actions = np.zeros((batch_size, seq_length), dtype=np.uint8)
    resets = np.zeros((batch_size, seq_length), dtype=np.uint8)
    gameids = np.zeros((batch_size, seq_length), dtype=np.int32)
    scores = np.zeros((batch_size, seq_length), dtype=np.int32)

    key_vals = [
        ("tty_chars", chars),
        ("tty_colors", colors),
        ("tty_cursor", cursors),
        ("timestamps", timestamps),
        ("done", resets),
        ("gameids", gameids),
    ]
    if ttyrec_version >= 2:
        key_vals.append(("keypresses", actions))
    if ttyrec_version >= 3:
        key_vals.append(("scores", scores))

    # Load initial gameids.
    converters = [
        converter.Converter(rows, cols, ttyrec_version) for _ in range(batch_size)
    ]
    assert all(load_fn(c) for c in converters), "Not enough ttyrecs to fill a batch!"

    # Convert (at least one minibatch)
    _convert_frames = partial(convert_frames, load_fn=load_fn)
    gameids[0, -1] = 1  # basically creating a "do-while" loop by setting an indicator
    while np.any(
        gameids[:, -1] != 0
    ):  # loop until only padding is found, i.e. end of data
        list(
            map_fn(
                _convert_frames,
                converters,
                chars,
                colors,
                cursors,
                timestamps,
                actions,
                scores,
                resets,
                gameids,
            )
        )

        yield dict(key_vals)


class TtyrecDataset:
    """Dataset object to allow iteration through the ttyrecs found in our ttyrec
    database.
    """

    def __init__(
        self,
        dataset_name,
        batch_size=128,
        seq_length=32,
        rows=24,
        cols=80,
        dbfilename=nld.db.DB,
        threadpool=None,
        gameids=None,
        shuffle=True,
        loop_forever=False,
        subselect_sql=None,
        subselect_sql_args=None,
    ):
        """
        An iterable dataset to load minibatches of NetHack games from compressed
        ttyrec*.bz2 files into numpy arrays. (shape: [batch_size, seq_length, ...])

        This class makes use of a sqlite3 database at `dbfilename` to find the
        metadata and the location of files in a dataset. It then uses these to
        create generators which convert the ttyrecs on the fly. Note that the
        dataset generators always reuse their numpy arrays, writing into the
        arrays instead of generating new ones. Methods to create and populate the
        db from an NLE directry can be found in `populate_db.py`.

        Example
        -------
            ```
            import nle.dataset as nld

            if not os.path.exists(nld.db.DB):
                nld.db.create()
                nld.populate_db.add_nledata_directory('path/to/nle_data', "data1")

            dataset = nld.TtyrecDataset("data1"):

            for mb in dataset:
                # NB: dataset reuses np arrays, for performance reasons
                print(mb)
            ```

        :param batch_size: Number of parallel games to load.
        :param seq_length: Number of frames to load per game.
        :param rows: Row size of the terminal screen.
        :param cols: Column size of the terminal screen.
        :param dbfilename: Path to the database file
        :param gameids: Use a subselection of games (gameids) only.
        :param shuffle: Shuffle the order of gameids before iterating through them.
        :param loop_forever: If true, cycle through gameids forever,
            insted of padding empty batch dims with 0's.
        :param subselect_sql: SQL Query to subselect games (gameids) using metadata
        :param subselect_sql_args: SQL Query Args to subselect games (gameids)
            using metadata.
        """
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.rows = rows
        self.cols = cols

        self.shuffle = shuffle
        self.subselect_sql = subselect_sql
        self.loop_forever = loop_forever

        sql_args = (dataset_name,)
        core_sql = """
            SELECT ttyrecs.gameid, ttyrecs.part, ttyrecs.path
            FROM ttyrecs
            INNER JOIN datasets ON ttyrecs.gameid=datasets.gameid
            WHERE datasets.dataset_name=?"""

        meta_sql = """
            SELECT games.*
            FROM games
            INNER JOIN datasets ON games.gameid=datasets.gameid
            WHERE datasets.dataset_name=?"""

        if subselect_sql:
            path_select = """
                SELECT ttyrecs.gameid, ttyrecs.part, ttyrecs.path
                FROM ttyrecs
                INNER JOIN datasets ON ttyrecs.gameid=datasets.gameid
                WHERE datasets.dataset_name=?
                AND ttyrecs.gameid IN (%s)"""
            core_sql = path_select % subselect_sql

            meta_select = """
                SELECT games.*
                FROM games
                INNER JOIN datasets ON games.gameid=datasets.gameid
                WHERE datasets.dataset_name=?
                AND games.gameid IN (%s)"""
            meta_sql = meta_select % subselect_sql
            sql_args = subselect_sql_args if subselect_sql_args else tuple()
            sql_args = (dataset_name,) + sql_args

        self._games = defaultdict(list)
        self._meta = None  # Populate lazily.
        self.dbfilename = dbfilename
        with nld.db.connect(self.dbfilename) as conn:
            c = conn.cursor()

            for row in c.execute(core_sql, sql_args):
                self._games[row[0]].append(row[1:3])

            # Guarantee order is [part0, ..., partN] for multi-part games.
            for files in self._games.values():
                files.sort()

            self._rootpath = nld.db.get_root(dataset_name, conn)
            self._ttyrec_version = nld.db.get_ttyrec_version(dataset_name, conn)

        if gameids is None:
            gameids = self._games.keys()

        self._core_sql = core_sql
        self._meta_sql = meta_sql
        self._sql_args = sql_args
        self._gameids = list(gameids)
        self._threadpool = threadpool
        self._map = partial(self._threadpool.map, timeout=60) if threadpool else map

    def get_paths(self, gameid):
        return [path for _, path in self._games[gameid]]

    def get_meta(self, gameid):
        if self._meta is None:
            self.populate_metadata()
        if gameid not in self._meta:
            return None
        return self._meta[gameid][0]

    def get_meta_columns(self):
        if self._meta is None:
            self.populate_metadata()
        return self._meta_cols

    def populate_metadata(self):
        self._meta = defaultdict(list)
        with nld.db.connect(self.dbfilename) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            for row in c.execute(self._meta_sql, self._sql_args):
                self._meta[row[0]].append(row)
            self._meta_cols = [desc[0] for desc in c.description]

    def _make_load_fn(self, gameids):
        """Make a closure to load the next gameid from the db into the converter."""
        lock = threading.Lock()
        count = [0]

        def _load_fn(converter):
            """Take the next part of the current game if available, else new game.
            Return True if load successful, else False."""
            gameid = converter.gameid
            part = converter.part + 1

            files = self.get_paths(gameid)
            if gameid == 0 or part >= len(files):
                with lock:
                    i = count[0]
                    count[0] += 1

                if (not self.loop_forever) and i >= len(gameids):
                    return False

                gameid = gameids[i % len(gameids)]
                files = self.get_paths(gameid)
                part = 0

            filename = files[part]
            filepath = os.path.join(self._rootpath, filename)
            converter.load_ttyrec(filepath, gameid=gameid, part=part)
            return True

        return _load_fn

    def __iter__(self):
        gameids = list(self._gameids)
        if self.shuffle:
            np.random.shuffle(gameids)

        return _ttyrec_generator(
            self.batch_size,
            self.seq_length,
            self.rows,
            self.cols,
            self._make_load_fn(gameids),
            self._map,
            self._ttyrec_version,
        )

    def get_ttyrecs(self, gameids, chunk_size=None):
        """Fetch data from a single episode, chunked into a sequence of tensors."""
        seq_length = chunk_size or self.seq_length
        mbs = []
        for mb in _ttyrec_generator(
            len(gameids),
            seq_length,
            self.rows,
            self.cols,
            self._make_load_fn(gameids),
            self._map,
            self._ttyrec_version,
        ):
            mbs.append({k: t.copy() for k, t in mb.items()})
        return mbs

    def get_ttyrec(self, gameid, chunk_size=None):
        return self.get_ttyrecs([gameid], chunk_size)
