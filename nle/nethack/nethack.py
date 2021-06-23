# Copyright (c) Facebook, Inc. and its affiliates.
import os
import pkg_resources
import shutil
import tempfile

import numpy as np

from nle import _pynethack


DLPATH = os.path.join(os.path.dirname(_pynethack.__file__), "libnethack.so")

DUNGEON_SHAPE = (_pynethack.nethack.ROWNO, _pynethack.nethack.COLNO - 1)
BLSTATS_SHAPE = (_pynethack.nethack.NLE_BLSTATS_SIZE,)
MESSAGE_SHAPE = (_pynethack.nethack.NLE_MESSAGE_SIZE,)
PROGRAM_STATE_SHAPE = (_pynethack.nethack.NLE_PROGRAM_STATE_SIZE,)
INTERNAL_SHAPE = (_pynethack.nethack.NLE_INTERNAL_SIZE,)
MISC_SHAPE = (_pynethack.nethack.NLE_MISC_SIZE,)
INV_SIZE = (_pynethack.nethack.NLE_INVENTORY_SIZE,)
INV_STRS_SHAPE = (
    _pynethack.nethack.NLE_INVENTORY_SIZE,
    _pynethack.nethack.NLE_INVENTORY_STR_LENGTH,
)
SCREEN_DESCRIPTIONS_SHAPE = DUNGEON_SHAPE + (
    _pynethack.nethack.NLE_SCREEN_DESCRIPTION_LENGTH,
)
TERMINAL_SHAPE = (_pynethack.nethack.NLE_TERM_LI, _pynethack.nethack.NLE_TERM_CO)

OBSERVATION_DESC = {
    "glyphs": dict(shape=DUNGEON_SHAPE, dtype=np.int16),
    "chars": dict(shape=DUNGEON_SHAPE, dtype=np.uint8),
    "colors": dict(shape=DUNGEON_SHAPE, dtype=np.uint8),
    "specials": dict(shape=DUNGEON_SHAPE, dtype=np.uint8),
    "blstats": dict(shape=BLSTATS_SHAPE, dtype=np.int64),
    "message": dict(shape=MESSAGE_SHAPE, dtype=np.uint8),
    "program_state": dict(shape=PROGRAM_STATE_SHAPE, dtype=np.int32),
    "internal": dict(shape=INTERNAL_SHAPE, dtype=np.int32),
    "inv_glyphs": dict(shape=INV_SIZE, dtype=np.int16),
    "inv_letters": dict(shape=INV_SIZE, dtype=np.uint8),
    "inv_oclasses": dict(shape=INV_SIZE, dtype=np.uint8),
    "inv_strs": dict(shape=INV_STRS_SHAPE, dtype=np.uint8),
    "screen_descriptions": dict(shape=SCREEN_DESCRIPTIONS_SHAPE, dtype=np.uint8),
    "tty_chars": dict(shape=TERMINAL_SHAPE, dtype=np.uint8),
    "tty_colors": dict(shape=TERMINAL_SHAPE, dtype=np.int8),
    "tty_cursor": dict(shape=(2,), dtype=np.uint8),
    "misc": dict(shape=MISC_SHAPE, dtype=np.int32),
}


NETHACKOPTIONS = (
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
)

HACKDIR = os.getenv("HACKDIR", pkg_resources.resource_filename("nle", "nethackdir"))
WIZKIT_FNAME = "wizkit.txt"


def _set_env_vars(options, hackdir, wizkit=None):
    # TODO: Investigate not using environment variables for this.
    os.environ["NETHACKOPTIONS"] = ",".join(options)
    os.environ["HACKDIR"] = hackdir
    os.environ["TERM"] = "ansi"
    if wizkit is not None:
        os.environ["WIZKIT"] = wizkit


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
        ttyrec="nle.ttyrec.bz2",
        options=None,
        copy=False,
        wizard=False,
        hackdir=HACKDIR,
    ):
        self._copy = copy

        if not os.path.exists(hackdir) or not os.path.exists(
            os.path.join(hackdir, "sysconf")
        ):
            raise FileNotFoundError(
                "Couldn't find NetHack installation at '%s'." % hackdir
            )

        # Create a HACKDIR for us.
        self._tempdir = tempfile.TemporaryDirectory(prefix="nle")
        self._vardir = self._tempdir.name

        # Save cwd and restore later. Currently libnethack changes
        # directory on loading.
        self._oldcwd = os.getcwd()

        # Symlink a few files.
        for fn in ["nhdat", "sysconf"]:
            os.symlink(os.path.join(hackdir, fn), os.path.join(self._vardir, fn))
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
        self._wizard = wizard

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

    def _write_wizkit_file(self, wizkit_items):
        # TODO ideally we need to check the validity of the requested items
        with open(os.path.join(self._vardir, WIZKIT_FNAME), "w") as f:
            for item in wizkit_items:
                f.write("%s\n" % item)

    def reset(self, new_ttyrec=None, wizkit_items=None):
        if wizkit_items is not None:
            if not self._wizard:
                raise ValueError("Set wizard=True to use the wizkit option.")
            self._write_wizkit_file(wizkit_items)
            _set_env_vars(self._options, self._vardir, wizkit=WIZKIT_FNAME)
        else:
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
        try:
            os.chdir(self._oldcwd)
        except IOError:
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self._tempdir.cleanup()

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

    def how_done(self):
        return self._pynethack.how_done()
