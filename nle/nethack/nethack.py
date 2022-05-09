# Copyright (c) Facebook, Inc. and its affiliates.
import os
import shutil
import sys
import tempfile
import warnings
import weakref

import numpy as np
import pkg_resources

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
    "autopickup",
    "color",
    "disclose:+i +a +v +g +c +o",
    "mention_walls",
    "nobones",
    "nocmdassist",
    "nolegacy",
    "nosparkle",
    "pickup_burden:unencumbered",
    "pickup_types:$?!/",
    "runmode:teleport",
    "showexp",
    "showscore",
    "time",
)

HACKDIR = pkg_resources.resource_filename("nle", "nethackdir")
TTYREC_VERSION = 3


def _new_dl_linux(vardir):
    if hasattr(os, "memfd_create"):
        target = os.memfd_create("nle.so")
        path = "/proc/self/fd/%i" % target
        try:
            shutil.copyfile(DLPATH, path)  # Should use sendfile.
        except IOError:
            os.close(target)
            raise
        return os.fdopen(target), path

    # Otherwise, no memfd_create. Try with O_TMPFILE via the tempfile module.
    dl = tempfile.TemporaryFile(suffix="libnethack.so", dir=vardir)
    path = "/proc/self/fd/%i" % dl.fileno()
    shutil.copyfile(DLPATH, path)  # Should use sendfile.
    return dl, path


def _new_dl(vardir):
    """Creates a copied .so file to allow for multiple independent NLE instances"""
    if sys.platform == "linux":
        return _new_dl_linux(vardir)

    # MacOS has no memfd_create or O_TMPFILE. Using /dev/fd/{FD} as an argument
    # to dlopen doesn't work after unlinking from the file system. So let's copy
    # instead and hope vardir gets properly deleted at some point.
    dl = tempfile.NamedTemporaryFile(suffix="libnethack.so", dir=vardir)
    shutil.copyfile(DLPATH, dl.name)  # Might use fcopyfile.
    return dl, dl.name


def _close(pynethack, dl, tempdir, warn=True):
    if pynethack is not None:
        pynethack.close()
    if dl is not None:
        dl.close()
    if tempdir is not None:
        tempdir.cleanup()
    if warn:
        warnings.warn("nethack.Nethack instance not closed", ResourceWarning)


def tty_render(chars, colors, cursor=None):
    """Returns chars as string with ANSI escape sequences.

    Args:
      chars: A row x columns numpy array of chars.
      colors: A numpy array of colors (0-15), same shape as chars.
      cursor: An optional (row, column) index for the cursor,
        displayed as underlined.

    Returns:
      A string with chars decorated by ANSI escape sequences.
    """
    rows, cols = chars.shape
    if cursor is None:
        cursor = (-1, -1)
    cursor = tuple(cursor)
    result = ""
    for i in range(rows):
        result += "\n"
        for j in range(cols):
            entry = "\033[%d;3%dm%s" % (
                # & 8 checks for brightness.
                bool(colors[i, j] & 8),
                colors[i, j] & ~8,
                chr(chars[i, j]),
            )
            if cursor != (i, j):
                result += entry
            else:
                result += "\033[4m%s\033[0m" % entry
    return result + "\033[0m"


# TODO: Not thread-safe for many reasons.
class Nethack:
    _instances = 0

    def __init__(
        self,
        observation_keys=OBSERVATION_DESC.keys(),
        playername="Agent-mon-hum-neu-mal",
        ttyrec="nle.ttyrec%i.bz2" % TTYREC_VERSION,
        options=None,
        copy=False,
        wizard=False,
        hackdir=HACKDIR,
        spawn_monsters=True,
        scoreprefix="",
    ):
        self._copy = copy

        if not os.path.exists(hackdir) or not os.path.exists(
            os.path.join(hackdir, "nhdat")
        ):
            raise FileNotFoundError(
                "Couldn't find NetHack installation at '%s'." % hackdir
            )

        # Create a HACKDIR for us.
        self._tempdir = tempfile.TemporaryDirectory(prefix="nle")
        self._vardir = self._tempdir.name

        # Symlink a nhdat.
        os.symlink(os.path.join(hackdir, "nhdat"), os.path.join(self._vardir, "nhdat"))

        # Touch files, so lock_file() in files.c passes.
        for fn in ["perm", "record", "logfile"]:
            os.close(os.open(os.path.join(self._vardir, fn), os.O_CREAT))
        if scoreprefix:
            os.close(os.open(scoreprefix + "xlogfile", os.O_CREAT))
        else:
            os.close(os.open(os.path.join(self._vardir, "xlogfile"), os.O_CREAT))

        os.mkdir(os.path.join(self._vardir, "save"))

        # An assortment of hacks:
        #   Copy our .so into self._vardir to load several copies of the dl.
        #   (Or use a memfd_create hack to create a file that gets deleted on
        #    process exit.)
        self._dl, self.dlpath = _new_dl(self._vardir)

        # Finalize even when the rest of this constructor fails.
        self._finalizer = weakref.finalize(self, _close, None, self._dl, self._tempdir)

        if options is None:
            options = NETHACKOPTIONS
        self.options = list(options) + ["name:" + playername]
        if playername.split("-", 1)[1:] == ["@"]:
            # Random role. Unless otherwise specified, randomize
            # race/gender/alignment too.
            for key in ("race", "gender", "align"):
                if not any(o for o in options if o.startswith(key + ":")):
                    self.options.append("%s:random" % key)

        if wizard:
            self.options.append("playmode:debug")
        self._wizard = wizard
        self._nethackoptions = ",".join(self.options)
        if ttyrec is None:
            self._pynethack = _pynethack.Nethack(
                self.dlpath, self._vardir, self._nethackoptions, spawn_monsters
            )
        else:
            self._pynethack = _pynethack.Nethack(
                self.dlpath,
                ttyrec,
                self._vardir,
                self._nethackoptions,
                spawn_monsters,
                scoreprefix,
            )
        self._ttyrec = ttyrec

        self._finalizer.detach()
        self._finalizer = weakref.finalize(
            self, _close, self._pynethack, self._dl, self._tempdir
        )

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

    def reset(self, new_ttyrec=None, wizkit_items=None):
        if wizkit_items is not None:
            if not self._wizard:
                raise ValueError("Set wizard=True to use the wizkit option.")
            # TODO ideally we need to check the validity of the requested items
            self._pynethack.set_wizkit("\n".join(wizkit_items))
        if new_ttyrec is None:
            self._pynethack.reset()
        else:
            self._pynethack.reset(new_ttyrec)
            self._ttyrec = new_ttyrec
        # No seeding performed here: If we fixed the seeds, we'd only
        # get one episode.
        return self._step_return()

    def close(self):
        if self._finalizer.detach():
            _close(self._pynethack, self._dl, self._tempdir, warn=False)
        self._pynethack = None
        self._dl = None
        self._tempdir = None

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
