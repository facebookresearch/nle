# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import logging
import os
import shutil
import tempfile
import time
import warnings
import weakref
import zipfile

from . import ptyprocess
import zmq


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Import all flatbuffer modules.
    from nle.fbs import (  # noqa: F401
        Blstats,
        Condition,
        DLevel,
        Internal,
        InventoryItem,
        MenuItem,
        Message,
        NDArray,
        Observation,
        ProgramState,
        Seeds,
        Status,
        Window,
        You,
    )


SEED_KEYS = ["core", "disp"]

NETHACKOPTIONS = [
    "windowtype:rl",
    "color",
    "showexp",
    "autopickup",
    "pickup_types:$?!/",
    "pickup_burden:unencumbered",
]

HACKDIR = os.getenv("HACKDIR")

if HACKDIR is None:
    # Somewhat HACKy way of getting HACKDIR from installed nethack.
    script = shutil.which("nethack")
    if script is None:
        raise FileNotFoundError("Didn't find nethack in path. Is it installed?")
    with open(script, "r") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Could not determine HACKDIR from installed nethack")
            if line.startswith("HACKDIR="):
                HACKDIR = line[8:-1]  # Exclude newline.
                break


EXECUTABLE = os.path.join(HACKDIR, "nethack")
if not os.path.exists(EXECUTABLE):
    raise FileNotFoundError(
        "Couldn't run nethack in %s as file doesn't exist" % EXECUTABLE
    )


def _exec_nethack(playername, hackdir, seeds=None, options=NETHACKOPTIONS):
    """Turns current process into NetHack with right environment variables."""
    user = playername % {"pid": os.getpid()}

    env = {
        "HACKDIR": hackdir,
        "NETHACKOPTIONS": ",".join(options),
        "USER": user,
        "SHELL": "/bin/bash",
        "TERM": "xterm-256color",
    }

    if seeds is not None:
        for name, seed in seeds.items():
            name = "NLE_SEED_" + name.upper()
            env[name] = str(seed)

    command = EXECUTABLE + " -u" + user

    shell = os.environ.get("SHELL", "/bin/bash")
    os.execle(shell, os.path.basename(shell), "-c", command, env)
    raise OSError("This shouldn't happen.")


def _finalize_one_run(hackdir):
    prefix = str(os.getuid())
    if not os.path.exists(hackdir):
        # Removed in finalizer of NetHack object. Also fine.
        return

    with os.scandir(hackdir) as scandir:
        for entry in scandir:
            if entry.name.startswith(prefix):
                os.unlink(entry.path)


def _recordclosefn(archive, filename):
    archive.write(filename)
    os.unlink(filename)


class NetHack:
    def __init__(
        self,
        archivefile="nethack.%(pid)i.%(time)s.zip",
        playername="Agent%(pid)i-mon-hum-neu-mal",
        options=None,
        rows=24,
        columns=80,
        context=None,
    ):
        """Constructs a new NetHack environment."""
        self._playername = playername
        self._rows = rows
        self._columns = columns

        if options is None:
            options = NETHACKOPTIONS
        self._nethackoptions = options

        self._episode = 0
        self._info = {}

        self._finalizers = []

        if not os.path.exists(HACKDIR) or not os.path.exists(
            os.path.join(HACKDIR, "nethack")
        ):
            raise FileNotFoundError("Couldn't find NetHack installation.")

        # Create a HACKDIR for us.
        self._vardir = tempfile.mkdtemp(prefix="nle")

        os.symlink(os.path.join(HACKDIR, "nhdat"), os.path.join(self._vardir, "nhdat"))

        # touch a few files.
        for filename in ["sysconf", "perm", "logfile", "xlogfile"]:
            os.close(os.open(os.path.join(self._vardir, filename), os.O_CREAT))
        os.mkdir(os.path.join(self._vardir, "save"))

        if archivefile is None:
            self._archive = None
            self._recordclosefn = lambda f: None
        else:
            try:
                self._archive = zipfile.ZipFile(
                    archivefile
                    % {"pid": os.getpid(), "time": time.strftime("%Y%m%d-%H%M%S")},
                    "x",
                )
            except FileExistsError:
                logging.exception("Archive file %s exists, terminating" % archivefile)
                raise

            self._recordclosefn = functools.partial(_recordclosefn, self._archive)
            self._finalizers.append(
                # Cannot close archive before final call to _recordclosefn.
                # Binding to lifetime of self doesn't work here, so bind to
                # _recordclosefn itself.
                weakref.finalize(self._recordclosefn, self._archive.close)
            )
            logging.info("Archiving replays in %s" % self._archive.filename)

        self._context = context or zmq.Context.instance()

        self._finalizers.append(weakref.finalize(self, shutil.rmtree, self._vardir))

        self._exec_nethack = None
        self.seed(None)  # Sets self._exec_nethack.

        # TODO(heiner): Consider having a collections.deque of processes for
        # pre-loading. This requires us to have different connections, which
        # would be great to support more than one NetHack object.
        self._process = None

    def _recv(self):
        buf = self._socket.recv()
        message = Message.Message.GetRootAsMessage(buf, 0)
        # TODO(heiner): Consider waitpid'ing to get process status.
        return message, message.NotRunning()

    def reset(self):
        if self._archive is None:
            self.recordname = None
        else:
            self.recordname = "nethack.run.%i.%%(time)s.%%(pid)i.ttyrec" % self._episode
        self._process = ptyprocess.PtyProcess(
            target=self._exec_nethack,
            recordclosefn=self._recordclosefn,
            recordname=self.recordname,
        )
        self._process.fork(rows=self._rows, columns=self._columns, wait_for_output=True)

        socketfile = os.path.join(self._vardir, "%i.nle.sock" % self._process.pid)
        address = "ipc://" + socketfile
        logging.debug("Connecting to %s...", address)
        self._socket = self._context.socket(zmq.PULL)
        self._socket.connect(address)

        weakref.finalize(self._process, self._socket.close)
        weakref.finalize(self._process, _finalize_one_run, self._vardir)

        if not self._socket.poll(timeout=1000):  # 1s timeout.
            raise IOError("No response received from NetHack process -- is it running?")

        message, done = self._recv()
        assert not done, "NetHack closed without input."

        # Connection established, can remove socket file from file system.
        os.unlink(socketfile)

        self._info["pid"] = self._process.pid
        self._info["episode"] = self._episode

        self._episode += 1

        return message

    def step(self, action):
        self._process.write(bytes((action,)))
        message, done = self._recv()

        return message, done, self._info

    def close(self):
        del self._process  # Triggers finalizer.
        for f in self._finalizers:
            f()

    def seed(self, seeds):
        if isinstance(seeds, dict):
            for k in SEED_KEYS:
                if k not in seeds:
                    continue
                if seeds[k] < 1:
                    raise ValueError(
                        "Found %d for %s, but seeds must be positive integers.",
                        seeds[k],
                        k,
                    )
        elif seeds is not None:
            raise ValueError("`seeds` is %s, but must be either None or a dict.", seeds)

        self._seeds = seeds

        self._exec_nethack = functools.partial(
            _exec_nethack,
            self._playername,
            self._vardir,
            self._seeds,
            self._nethackoptions,
        )
