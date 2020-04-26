# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest
import tempfile

import numpy as np

from nle import nethack


def _fb_ndarray_to_np(fb_ndarray):
    result = fb_ndarray.DataAsNumpy()
    result = result.view(np.typeDict[fb_ndarray.Dtype()])
    result = result.reshape(fb_ndarray.ShapeAsNumpy().tolist())
    return result


class NetHackTest(unittest.TestCase):
    def test_run(self):
        archivefile = tempfile.mktemp(suffix="nethack_test", prefix=".zip")
        game = nethack.NetHack(archivefile=archivefile)

        response = game.reset()
        actions = [
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
        ]

        for action in actions:
            while not response.ProgramState().InMoveloop():
                response, done, info = game.step(nethack.MiscAction.MORE)

            response, done, info = game.step(action)
            if done:
                # Only the good die young.
                response = game.reset()

            obs = response.Observation()
            chars = _fb_ndarray_to_np(obs.Chars())
            glyphs = _fb_ndarray_to_np(obs.Glyphs())

            status = response.Blstats()
            x, y = status.CursX(), status.CursY()

            self.assertEqual(np.count_nonzero(chars == ord("@")), 1)
            self.assertEqual(chars[y, x], ord("@"))

            mon = nethack.glyph_to_mon(glyphs[y][x])
            self.assertEqual(mon.mname, "monk")
            self.assertEqual(mon.mlevel, 10)

            class_sym = nethack.mlet_to_class_sym(mon.mlet)
            self.assertEqual(class_sym.sym, "@")
            self.assertEqual(class_sym.explain, "human or elf")

        self.assertEqual(os.waitpid(info["pid"], os.WNOHANG), (0, 0))

        del game  # Should kill process.

        with self.assertRaisesRegex(OSError, "No (child|such)? process"):
            os.waitpid(info["pid"], 0)


class HelperTest(unittest.TestCase):
    def test_simple(self):
        glyph = 155  # Lichen.

        mon = nethack.glyph_to_mon(glyph)

        self.assertEqual(mon.mname, "lichen")

        cs = nethack.mlet_to_class_sym(mon.mlet)

        self.assertEqual(cs.sym, "F")
        self.assertEqual(cs.explain, "fungus or mold")

        self.assertEqual(nethack.NHW_MESSAGE, 1)
        self.assertTrue(hasattr(nethack, "MAXWIN"))


if __name__ == "__main__":
    unittest.main()
