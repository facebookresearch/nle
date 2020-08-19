import os
import pkg_resources
import shutil
import tempfile

import numpy as np

from nle import _pynethack


DLPATH = os.path.join(os.path.dirname(_pynethack.__file__), "libnethack.so")

# TODO: Consider getting this from C++.
DUNGEON_SHAPE = (21, 79)
BLSTATS_SHAPE = (_pynethack.nethack.NLE_BLSTATS_SIZE,)
MESSAGE_SHAPE = (256,)
PROGRAM_STATE_SHAPE = (5,)
INTERNAL_SHAPE = (_pynethack.nethack.NLE_INTERNAL_SIZE,)

OBSERVATION_DESC = {
    "glyphs": dict(shape=DUNGEON_SHAPE, dtype=np.int16),
    "chars": dict(shape=DUNGEON_SHAPE, dtype=np.uint8),
    "colors": dict(shape=DUNGEON_SHAPE, dtype=np.uint8),
    "specials": dict(shape=DUNGEON_SHAPE, dtype=np.uint8),
    "blstats": dict(shape=BLSTATS_SHAPE, dtype=np.int64),
    "message": dict(shape=MESSAGE_SHAPE, dtype=np.uint8),
    "program_state": dict(shape=PROGRAM_STATE_SHAPE, dtype=np.int32),
    "internal": dict(shape=INTERNAL_SHAPE, dtype=np.int32),
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

HACKDIR = os.getenv("HACKDIR", pkg_resources.resource_filename("nle", "nethackdir"))


def _set_env_vars(options, hackdir):
    # TODO: Investigate not using environment variables for this.
    os.environ["NETHACKOPTIONS"] = ",".join(options)
    os.environ["HACKDIR"] = hackdir
    os.environ["TERM"] = os.environ.get("TERM", "screen")


# TODO: Not thread-safe for many reasons.
# TODO: On Linux, we could use dlmopen to use different linker namespaces,
# which should allow several instances of this. On MacOS, that seems
# a tough call.
class Nethack:
    _instances = 0

    def __init__(
        self,
        observation_keys=OBSERVATION_DESC.keys(),
        playername="Agent-mon-hum-neu-mal",
        options=None,
        copy=False,
    ):
        self._copy = copy

        if not os.path.exists(HACKDIR) or not os.path.exists(
            os.path.join(HACKDIR, "sysconf")
        ):
            raise FileNotFoundError("Couldn't find NetHack installation.")

        # Create a HACKDIR for us.
        self._vardir = tempfile.mkdtemp(prefix="nle")
        os.symlink(os.path.join(HACKDIR, "nhdat"), os.path.join(self._vardir, "nhdat"))
        # touch a few files.
        for filename in ["sysconf", "perm", "logfile", "xlogfile"]:
            os.close(os.open(os.path.join(self._vardir, filename), os.O_CREAT))
        os.mkdir(os.path.join(self._vardir, "save"))

        # Hacky AF: Copy our so into this directory to load several copies ...
        dlpath = os.path.join(self._vardir, "libnethack.so")
        shutil.copyfile(DLPATH, dlpath)

        if options is None:
            options = NETHACKOPTIONS
        self._options = list(options) + ["name:" + playername]

        _set_env_vars(self._options, self._vardir)
        self._seeds = None

        self._pynethack = _pynethack.Nethack(dlpath)

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
        _set_env_vars(self._options, self._vardir)
        self._pynethack.reset()
        if self._seeds is not None:
            self._pynethack.set_seed(*self._seeds)
        return self._step_return()

    def close(self):
        self._pynethack.close()

    def seed(self, core, disp, reseed=False):
        if core is None or disp is None:
            self._seeds = None
            return
        self._pynethack.set_seed(core, disp, reseed)
        self._seeds = (core, disp, reseed)
