#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import fcntl
import os
import signal
import struct
import termios
import time
import tty

parser = argparse.ArgumentParser()
parser.add_argument(
    "-1",
    "--no_input",
    action="store_true",
    help=("Save in ttyrec (not ttyrec2) format, " "without input data"),
)
parser.add_argument(
    "-e", "--execute", metavar="CMD", help="command to run (default: /bin/sh)"
)
parser.add_argument(
    "-a",
    "--append",
    action="store_true",
    help="append instead of truncating output file",
)
parser.add_argument("--keep_stderr", action="store_true", help="don't process stderr")
parser.add_argument(
    "filename", default="out.ttyrec", type=str, nargs="?", help="tty record file"
)
parser.add_argument("-c", "--columns", type=int, help="override number of columns")
parser.add_argument("-r", "--rows", type=int, help="override number rows")


def write_header(fp, length, channel):
    try:
        usec = time.time_ns() // 1000
    except AttributeError:  # time_ns is only available in Python 3.7+.
        usec = int(1000 * 1000 * time.time())
    sec = usec // 1000 // 1000
    usec -= 1000 * 1000 * sec  # now = sec + usec * 1e-6

    if FLAGS.no_input:
        os.write(fp, struct.pack("<iii", sec, usec, length))
    else:
        os.write(fp, struct.pack("<iiiB", sec, usec, length, channel))


def dooutput(fdmaster, fdslave, script):
    os.close(0)
    os.close(fdslave)

    while True:
        buf = os.read(fdmaster, 1024)
        if not buf:
            break
        os.write(1, buf)
        fcntl.flock(script, fcntl.LOCK_EX)
        write_header(script, len(buf), 0)
        os.write(script, buf)
        fcntl.flock(script, fcntl.LOCK_UN)

    os.close(fdmaster)
    os._exit(0)


def doshell(fdmaster, fdslave):
    os.setsid()
    fcntl.ioctl(fdslave, termios.TIOCSCTTY)

    os.close(fdmaster)
    os.dup2(fdslave, 0)
    os.dup2(fdslave, 1)
    if not FLAGS.keep_stderr:
        os.dup2(fdslave, 2)
    os.close(fdslave)

    shell = os.environ.get("SHELL", "/bin/bash")

    if FLAGS.execute:
        os.execl(shell, os.path.basename(shell), "-c", FLAGS.execute)
    else:
        os.execl(shell, os.path.basename(shell))


def doinput(fdmaster, fdslave, script):
    while True:
        buf = os.read(0, 1024)
        if not buf:
            break
        os.write(fdmaster, buf)

        if FLAGS.no_input:
            continue

        fcntl.flock(script, fcntl.LOCK_EX)
        write_header(script, len(buf), 1)
        os.write(script, buf)
        fcntl.flock(script, fcntl.LOCK_UN)


def main():
    global FLAGS
    FLAGS = parser.parse_args()

    flags = os.O_CREAT | os.O_WRONLY
    if FLAGS.append:
        flags |= os.O_APPEND
    else:
        flags |= os.O_TRUNC
    scriptfd = os.open(FLAGS.filename, flags)
    os.close(scriptfd)

    winsz = bytearray(8)
    fcntl.ioctl(0, termios.TIOCGWINSZ, winsz)

    ws_row, ws_col, ws_xpixel, ws_ypixel = struct.unpack("HHHH", winsz)
    if FLAGS.rows is not None:
        ws_row = FLAGS.rows
    if FLAGS.columns is not None:
        ws_col = FLAGS.columns
    struct.pack_into("HHHH", winsz, 0, ws_row, ws_col, ws_xpixel, ws_ypixel)

    fdmaster, fdslave = os.openpty()
    fcntl.ioctl(fdslave, termios.TIOCSWINSZ, winsz)

    tt = termios.tcgetattr(0)

    tty.setraw(0)
    rtt = termios.tcgetattr(0)
    rtt[3] &= ~termios.ECHO  # lflags
    termios.tcsetattr(0, termios.TCSAFLUSH, rtt)

    scriptfd = 0
    child = 0
    subchild = 0

    def finish(signum, frame):
        pid, status = os.waitpid(-1, os.WNOHANG)
        if pid == child:
            termios.tcsetattr(0, termios.TCSAFLUSH, tt)

        os.close(scriptfd)
        os._exit(0)

    signal.signal(signal.SIGCHLD, finish)

    child = os.fork()

    if child == 0:
        subchild = os.fork()

        if subchild != 0:
            scriptfd = os.open(FLAGS.filename, os.O_APPEND | os.O_WRONLY)
            dooutput(fdmaster, fdslave, scriptfd)
        else:
            doshell(fdmaster, fdslave)
    try:
        scriptfd = os.open(FLAGS.filename, os.O_APPEND | os.O_WRONLY)
        doinput(fdmaster, fdslave, scriptfd)
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)


if __name__ == "__main__":
    main()
