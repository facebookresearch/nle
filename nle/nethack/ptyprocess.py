# Copyright (c) Facebook, Inc. and its affiliates.
import fcntl
import os
import signal
import struct
import termios
import threading
import time
import weakref


def _kill(fd, pid, thread):
    os.kill(pid, signal.SIGTERM)
    os.close(fd)
    thread.join()
    os.waitpid(pid, 0)


def _write_frame(fd, buf, channel=0):
    """Write one frame in ttyrec2 format to fd."""
    try:
        usec = time.time_ns() // 1000
    except AttributeError:  # time_ns is only available in Python 3.7+.
        usec = int(1000 * 1000 * time.time())
    sec = usec // 1000 // 1000
    usec -= 1000 * 1000 * sec  # now = sec + usec * 1e-6
    os.write(fd, struct.pack("<iiiB", sec, usec, len(buf), channel))
    os.write(fd, buf)


def _thread_target(fd, record, lock):
    try:
        while True:
            buf = os.read(fd, 1024)
            if not buf:
                break
            with lock:
                _write_frame(record, buf)
    except IOError:
        pass
    os.close(record)


class PtyProcess:
    def __init__(
        self,
        target,
        recordname="process.%(time)s.%(pid)i.ttyrec",
        recordclosefn=lambda rn: None,
    ):
        self._target = target
        self.pid = None
        self.fd = None
        self._recordname = recordname
        self._recordfd = None
        self._recordclosefn = recordclosefn
        self._finalizer = None
        self._lock = threading.Lock()

    def write_record_frame(self, buf, channel):
        with self._lock:
            _write_frame(self._recordfd, buf, channel)

    def write(self, buf):
        self.write_record_frame(buf, 1)
        os.write(self.fd, buf)

    def fork(self, rows=40, columns=80, wait_for_output=False):
        self.pid, self.fd = os.forkpty()
        if self.pid == 0:
            # Both stdout and stderr are attached to fd. We won't be seeing errors.
            # Could be fixed with
            #   fdmaster, fdslave = os.openpty()
            #   # Fork & in child:
            #   os.close(fdmaster)
            #   os.setsid()
            #   fcntl.ioctl(fdslave, termios.TIOCSCTTY)
            #   os.dup2(fdslave, 0)
            #   os.dup2(fdslave, 1)
            self._target()
            os._exit(1)
        fcntl.ioctl(
            self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0)
        )
        if self._recordname is None:
            self.filename = None
            self._recordfd = os.open(os.devnull, os.O_WRONLY)
        else:
            self.filename = self._recordname % {
                "time": time.strftime("%Y%m%d-%H%M%S"),
                "pid": self.pid,
            }
            self._recordfd = os.open(
                self.filename, os.O_WRONLY | os.O_CREAT, mode=0o644
            )

        if wait_for_output:
            # Wait for child process to output something before returning.
            buf = os.read(self.fd, 1024)
            if not buf:
                raise RuntimeError("Could not read from child process")
            _write_frame(self._recordfd, buf)

        # Need to be careful not to keep any reference to self.
        thread = threading.Thread(
            target=_thread_target,
            args=(self.fd, self._recordfd, self._lock),
            daemon=True,  # Daemon so we can join in finalizer when exception thrown.
        )
        weakref.finalize(thread, self._recordclosefn, self.filename)
        thread.start()
        self._finalizer = weakref.finalize(self, _kill, self.fd, self.pid, thread)

    def term(self):
        if self._finalizer is not None:
            self._finalizer()
