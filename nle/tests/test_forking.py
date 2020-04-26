# Copyright (c) Facebook, Inc. and its affiliates.
import multiprocessing as mp
import unittest

from nle import nethack
import zmq


NUM_SUBPROCESSES = 2


def _run_nethack():
    env = nethack.NetHack(archivefile=None)
    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)


def _run_nethack_in_subprocesses(num_procs=1):
    ctx = mp.get_context("fork")

    processes = []

    for _ in range(num_procs):
        process = ctx.Process(target=_run_nethack)
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    return [process.exitcode for process in processes]


class ForkingNetHackTest(unittest.TestCase):
    def test_forking_without_nethack_in_parent(self, num_procs=NUM_SUBPROCESSES):
        self.assertEqual(_run_nethack_in_subprocesses(num_procs), [0] * num_procs)

    def test_forking_with_nethack_in_parent_new_context(
        self, num_procs=NUM_SUBPROCESSES
    ):
        env = nethack.NetHack(archivefile=None, context=zmq.Context())  # noqa: F841
        self.assertEqual(_run_nethack_in_subprocesses(num_procs), [0] * num_procs)

    def test_forking_with_nethack_in_parent(self, num_procs=NUM_SUBPROCESSES):
        # Breaks for pyzmq <= 18.0.0, fixed in
        # https://github.com/zeromq/pyzmq/commit/28c2a36836fc45c09ede4d9962498db449b642d1 # noqa
        env = nethack.NetHack(archivefile=None)  # noqa: F841
        self.assertEqual(_run_nethack_in_subprocesses(num_procs), [0] * num_procs)


if __name__ == "__main__":
    unittest.main()
