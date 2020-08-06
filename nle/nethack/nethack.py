import os

import numpy as np

from nle import _pynethack


DLPATH = os.path.join(os.path.dirname(_pynethack.__file__), "libnethack.so")

DUNGEON_SHAPE = (21, 79)
BLSTATS_SHAPE = (23,)
PROGRAM_STATE_SHAPE = (5,)

OBSERVATION_DESC = {
    "glyphs": dict(shape=DUNGEON_SHAPE, dtype=np.int16),
    "chars": dict(shape=DUNGEON_SHAPE, dtype=np.uint8),
    "colors": dict(shape=DUNGEON_SHAPE, dtype=np.uint8),
    "specials": dict(shape=DUNGEON_SHAPE, dtype=np.uint8),
    "blstats": dict(shape=BLSTATS_SHAPE, dtype=np.int64),
    "program_state": dict(shape=PROGRAM_STATE_SHAPE, dtype=np.int32),
}


NETHACKOPTIONS = [
    "color",
    "showexp",
    "autopickup",
    "pickup_types:$?!/",
    "pickup_burden:unencumbered",
    "nolegacy",
    "nocmdassist",
]

# TODO: Ensure we only create one of these, for now.
class Nethack:
    def __init__(
        self,
        observation_keys=OBSERVATION_DESC.keys(),
        playername="Agent-mon-hum-neu-mal",
        options=None,
        copy=False,
    ):
        self._copy = copy

        if options is None:
            options = NETHACKOPTIONS
        options = list(options) + ["name:" + playername]

        # TODO: Investigate not using environment variables for this.
        os.environ["NETHACKOPTIONS"] = ",".join(options)
        self._pynethack = _pynethack.Nethack(DLPATH)

        self._obs_buffers = {}

        for key in observation_keys:
            if key not in OBSERVATION_DESC:
                raise ValueError("Unknown observation '%s'" % key)
            self._obs_buffers[key] = np.zeros(**OBSERVATION_DESC[key])

        self._pynethack.set_buffers(**self._obs_buffers)

        self._obs = tuple(self._obs_buffers[key] for key in observation_keys)
        if self._copy:
            self._step_return = lambda: tuple(o.copy() for o in self._obs)
        else:
            self._step_return = lambda: self._obs

    def step(self, action):
        self._pynethack.step(action)
        return self._step_return(), self._pynethack.done()

    def reset(self):
        self._pynethack.reset()
        return self._step_return()

    def close(self):
        self._pynethack.close()
