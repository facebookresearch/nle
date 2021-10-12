# Copyright (c) Facebook, Inc. and its affiliates.

import os
import pickle

import numpy as np
import pkg_resources

from nle.tiles import MAXOTHTILE
from nle.tiles import glyph2tile


class GlyphMapper:
    """This class is used to map glyphs to rgb pixels."""

    def __init__(self):
        self.tiles = self.load_tiles()

    def load_tiles(self):
        """This function expects that tile.npy already exists.
        If it doesn't, call make_tiles.py in win/
        """

        tile_rgb_path = os.path.join(
            pkg_resources.resource_filename("nle", "tiles"),
            "tiles.pkl",
        )

        return pickle.load(open(tile_rgb_path, "rb"))

    def glyph_id_to_rgb(self, glyph_id):
        tile_id = glyph2tile[glyph_id]
        assert 0 <= tile_id <= MAXOTHTILE
        return self.tiles[tile_id]

    def _glyph_to_rgb(self, glyphs):
        # Expects glhyphs as two-dimensional numpy ndarray
        cols = None
        col = None

        for i in range(glyphs.shape[1]):
            for j in range(glyphs.shape[0]):
                rgb = self.glyph_id_to_rgb(glyphs[j, i])
                if col is None:
                    col = rgb
                else:
                    col = np.concatenate((col, rgb))

            if cols is None:
                cols = col
            else:
                cols = np.concatenate((cols, col), axis=1)
            col = None

        return cols

    def to_rgb(self, glyphs):
        return self._glyph_to_rgb(glyphs)
