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
PROGRAM_STATE_SHAPE = (_pynethack.nethack.NLE_PROGRAM_STATE_SIZE,)
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
    "nobones",
    "nolegacy",
    "nocmdassist",
    "disclose:+i +a +v +g +c +o",
    "runmode:teleport",
    "mention_walls",
    "nosparkle",
    "showexp",
    "showscore",
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
        ttyrec="nle.ttyrec",
        options=None,
        copy=False,
        wizard=False,
    ):
        self._copy = copy

        if not os.path.exists(HACKDIR) or not os.path.exists(
            os.path.join(HACKDIR, "sysconf")
        ):
            raise FileNotFoundError("Couldn't find NetHack installation.")

        # Create a HACKDIR for us.
        self._vardir = tempfile.mkdtemp(prefix="nle")

        # Symlink a few files.
        for fn in ["nhdat", "sysconf"]:
            os.symlink(os.path.join(HACKDIR, fn), os.path.join(self._vardir, fn))
        # Touch a few files.
        for fn in ["perm", "logfile", "xlogfile"]:
            os.close(os.open(os.path.join(self._vardir, fn), os.O_CREAT))
        os.mkdir(os.path.join(self._vardir, "save"))

        # Hacky AF: Copy our so into this directory to load several copies ...
        dlpath = os.path.join(self._vardir, "libnethack.so")
        shutil.copyfile(DLPATH, dlpath)

        if options is None:
            options = NETHACKOPTIONS
        self._options = list(options) + ["name:" + playername]
        if wizard:
            self._options.append("playmode:debug")

        _set_env_vars(self._options, self._vardir)
        self._ttyrec = ttyrec

        self._pynethack = _pynethack.Nethack(dlpath, ttyrec)

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

    def reset(self, new_ttyrec=None):
        _set_env_vars(self._options, self._vardir)
        if new_ttyrec is None:
            self._pynethack.reset()
        else:
            self._pynethack.reset(new_ttyrec)
            self._ttyrec = new_ttyrec
        # No seeding performed here: If we fixed the seeds, we'd only
        # get one episode.
        return self._step_return()

    def close(self):
        self._pynethack.close()

    def set_initial_seeds(self, core, disp, reseed=False):
        self._pynethack.set_initial_seeds(core, disp, reseed)

    def set_current_seeds(self, core=None, disp=None, reseed=False):
        """Sets the seeds of NetHack right now.

        If either of the three arguments is None, its current value will be
        used instead. Calling with default arguments will not change
        the RNG seeds but disable reseeding. Calling
        `set_current_seeds(reseed=None)` is a no-op and only returns the current
        values. If NetHack detects a good source of random numbers (true on most
        modern systems), its default value for `reseed` before the first call to
        this method is `True`.

        Arguments:
            core [int or None]: Seed for the core RNG.
            disp [int or None]: Seed for the disp (anti-TAS) RNG.
            reseed [boolean or None]: As an Anti-TAS (automation) measure,
                NetHack 3.6 reseeds with true randomness every now and then. This
                flag enables or disables this behavior. If set to True, trajectories
                won't be reproducible.

        Returns:
            [list] the seeds used by NetHack.
        """
        seeds = [core, disp, reseed]
        if any(s is None for s in seeds):
            if all(s is None for s in seeds):
                return
            for i, (s, s0) in enumerate(zip(seeds, self.get_current_seeds())):
                if s is None:
                    seeds[i] = s0
            return self._pynethack.set_seeds(*seeds)
        return self._pynethack.set_seeds(core, disp, reseed)

    def get_current_seeds(self):
        return self._pynethack.get_seeds()

    def in_normal_game(self):
        return self._pynethack.in_normal_game()
