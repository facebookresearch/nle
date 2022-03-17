import logging
import os
import sys
import threading
from collections import defaultdict
from functools import partial

import numpy as np
import torch

import nle.dataset.converter as converter
import nle.dataset.db as db

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def convert_frames(
    converter, chars, colors, curs, timestamps, actions, resets, gameids, load_fn
):
    """Convert frames for a single batch entry.

    :param converter: A ttychars.Converter object with a loaded file.
    :param chars: Array of characters -  np.array(np.uint8) [ SEQ x ROW x COL]
    :param colors: Array of colors -  np.array(np.uint8) [ SEQ x ROW x COL]
    :param curs: Array of cursors -  np.array(np.int16) [ SEQ x 2 ]
    :param timestamps: Array of timestamps -  np.array(np.int64) [ SEQ ]
    :param actions: Array of actions at t in response to output at t
        - np.array(np.uint8) [ SEQ ]
    :param resets: Array of resets -  np.array(np.uint8) [ SEQ ]
    :param gameids: Array of the gameid of each frame - np.array(np.int32) [ SEQ ]
    :param load_fn: A callback that loads the next file into a converter:
        sig: load_fn(converter) -> bool is_success

    Note actions will only be non-null if the ttyrec read_actions flag is set.
    """

    resets[0] = 0
    while True:
        remaining = converter.convert(chars, colors, curs, timestamps, actions)
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
            resets.fill(0)
            gameids.fill(0)
            return


def _ttyrec_generator(
    batch_size, seq_length, rows, cols, load_fn, map_fn, read_actions
):
    """A generator to fill minibatches with ttyrecs.

    :param load_fn: a function to load the next ttyrec into a converter.
       load_fn(ttyrecs.Converter conv) -> bool is_success
    :param map_fn: a function that maps a series of iterables through a fn.
       map_fn(fn, *iterables) -> <generator> (can use built-in map)

    """
    # Instantiate tensors and pin.
    chars = torch.zeros((batch_size, seq_length, rows, cols), dtype=torch.uint8)
    colors = torch.zeros((batch_size, seq_length, rows, cols), dtype=torch.int8)
    cursors = torch.zeros((batch_size, seq_length, 2), dtype=torch.int16)
    timestamps = torch.zeros((batch_size, seq_length), dtype=torch.int64)
    actions = torch.zeros((batch_size, seq_length), dtype=torch.uint8)
    resets = torch.zeros((batch_size, seq_length), dtype=torch.uint8)
    gameids = torch.zeros((batch_size, seq_length), dtype=torch.int32)

    npchars = chars.numpy()
    npcolors = colors.numpy()
    npcursors = cursors.numpy()
    nptimestamps = timestamps.numpy()
    npactions = actions.numpy()
    npresets = resets.numpy()
    npgameids = gameids.numpy()

    if torch.cuda.is_available():
        for tensor in [chars, colors, cursors, timestamps, actions, resets]:
            tensor.pin_memory()

    # Load initial gameids.
    converters = [
        converter.Converter(rows, cols, read_inputs=read_actions)
        for _ in range(batch_size)
    ]
    assert all(load_fn(c) for c in converters), "Not enough ttyrecs to fill a batch!"

    # Convert (at least one minibatch)
    _convert_frames = partial(convert_frames, load_fn=load_fn)
    npgameids[0, -1] = 1  # basically creating a "do-while" loop by setting an indicator
    while np.any(
        npgameids[:, -1] != 0
    ):  # loop until only padding is found, i.e. end of data
        list(
            map_fn(
                _convert_frames,
                converters,
                npchars,
                npcolors,
                npcursors,
                nptimestamps,
                npactions,
                npresets,
                npgameids,
            )
        )

        key_vals = [
            ("tty_chars", chars),
            ("tty_colors", colors),
            ("tty_cursors", cursors),
            ("timestamps", timestamps),
            ("done", resets.bool()),
            ("gameids", gameids),
        ]
        if read_actions:
            key_vals.append(("actions", actions))

        yield dict(key_vals)


class TtyrecDataset(torch.utils.data.IterableDataset):
    """Dataset object to allow iteration through the ttyrecs found in our ttyrec
    database.
    """

    def __init__(
        self,
        dataset_name,
        batch_size=128,
        seq_length=100,
        rows=24,
        cols=80,
        dbfilename=db.DB,
        threadpool=None,
        gameids=None,
        shuffle=False,
        read_actions=False,
        sql_subset=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.rows = rows
        self.cols = cols
        self.read_actions = read_actions

        self.shuffle = shuffle
        self.sql_subset = sql_subset

        core_sql = """
            SELECT ttyrecs.gameid, ttyrecs.part, ttyrecs.path
            FROM ttyrecs
            INNER JOIN datasets ON ttyrecs.gameid=datasets.gameid
            WHERE datasets.dataset_name=?"""

        if sql_subset is None:
            self.sql_subset = core_sql

        self._games = defaultdict(list)
        self._meta = defaultdict(list)
        with db.connect(dbfilename) as conn:
            c = conn.cursor()

            for row in c.execute(self.sql_subset, (dataset_name,)).fetchall():
                # A row is made up of [ gameid, part, path, meta1, meta2... metaN].
                # if row[0] in gameids:
                self._games[row[0]].append(row[1:3])
                self._meta[row[0]].append(row[3:])

            # Guarantee order is [part0, ..., partN] for multi-part games.
            for files in self._games.values():
                files.sort()

            self._meta_cols = [desc[0] for desc in c.description]
            self._rootpath = db.getroot(dataset_name, conn)

        if gameids is None:
            gameids = self._games.keys()

        self._gameids = list(gameids)
        self._threadpool = threadpool
        self._map = partial(self._threadpool.map, timeout=60) if threadpool else map

    def get_paths(self, gameid):
        return [path for _, path in self._games[gameid]]

    def get_meta(self, gameid):
        return self._meta[gameid]

    def get_meta_columns(self):
        return self._meta_cols

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

                if i >= len(gameids):
                    return False

                gameid = gameids[i]
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
            self.read_actions,
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
            self.read_actions,
        ):
            mbs.append({k: t.clone().detach() for k, t in mb.items()})
        return mbs

    def get_ttyrec(self, gameid, chunk_size=None):
        return self.get_ttyrecs([gameid], chunk_size)


def add_directory(path, name, filename=db.DB):
    if not os.path.isfile(filename):
        db.create(filename)
    # db.add_nledata_directory(path, dataset_name, filename)
    db.add_altorg_directory(path, dataset_name, filename)


if __name__ == "__main__":
    path = "/private/home/ehambro/fair/workspaces/autoascend-submission/nle_data"
    path = "/scratch/ehambro/altorg/altorg/111720"
    dataset_name = "altorg"
    dataset = add_directory(path, dataset_name)

    logging.info("%s" % db.countgames(dataset_name))
    dataset = TtyrecDataset(dataset_name)
    logging.info("%s" % len(dataset._gameids))
