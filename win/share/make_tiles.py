# Copyright (c) Facebook, Inc. and its affiliates.

import os
import pickle
import re

import numpy as np
import pkg_resources

FILES = ["monsters.txt", "objects.txt", "other.txt"]
DIR = os.path.dirname(os.path.realpath(__file__))
PATTERN = re.compile(r"^# tile (\d+) \((.*?)\)$", re.MULTILINE)
OUT_FILE = os.path.join(
    pkg_resources.resource_filename("nle", "tiles"),
    "tiles.pkl",
)

char2rgb = dict()

char2rgb["."] = (71, 108, 108)
char2rgb["A"] = (0, 0, 0)
char2rgb["B"] = (0, 182, 255)
char2rgb["C"] = (255, 108, 0)
char2rgb["D"] = (255, 0, 0)
char2rgb["E"] = (0, 0, 255)
char2rgb["F"] = (0, 145, 0)
char2rgb["G"] = (108, 255, 0)
char2rgb["H"] = (255, 255, 0)
char2rgb["I"] = (255, 0, 255)
char2rgb["J"] = (145, 71, 0)
char2rgb["K"] = (204, 79, 0)
char2rgb["L"] = (255, 182, 145)
char2rgb["M"] = (237, 237, 237)
char2rgb["N"] = (255, 255, 255)
char2rgb["O"] = (215, 215, 215)
char2rgb["P"] = (108, 145, 182)
char2rgb["Q"] = (18, 18, 18)
char2rgb["R"] = (54, 54, 54)
char2rgb["S"] = (73, 73, 73)
char2rgb["T"] = (82, 82, 82)
char2rgb["U"] = (205, 205, 205)
char2rgb["V"] = (104, 104, 104)
char2rgb["W"] = (131, 131, 131)
char2rgb["X"] = (140, 140, 140)
char2rgb["Y"] = (149, 149, 149)
char2rgb["Z"] = (195, 195, 195)
char2rgb["0"] = (100, 100, 100)
char2rgb["1"] = (72, 108, 108)


# Tile to numpyer ndarray
def tile_to_arr(tile):
    lines = [list(line) for line in tile.split("\n") if len(line) > 0]
    x = max(len(line) for line in lines)
    new_lines = [line + [" "] * (x - len(line)) for line in lines]
    return np.array(new_lines)


# Numpy ndarray to rgb array using color dict above
def arr_to_rgb(sym_arr):
    h, w = sym_arr.shape
    rgb_arr = np.ndarray((h, w, 3))

    for ix, iy in np.ndindex(sym_arr.shape):
        rgb_arr[ix, iy] = char2rgb[sym_arr[ix, iy]]

    return rgb_arr.astype(np.uint8)


def parse_tiles():
    id_to_rgb = {}
    tile_id = 0

    for fn in FILES:
        print("Reading {}".format(fn))
        with open(os.path.join(DIR, fn)) as f:
            lines = f.readlines()

        i = 0
        max_line = len(lines)
        while i < max_line:
            line = lines[i]
            if re.match(PATTERN, line):
                tile = "".join(lines[i + 2 : i + 18])
                tile = tile.replace(" ", "")
                tile_arr = tile_to_arr(tile)
                rgb = arr_to_rgb(tile_arr)
                id_to_rgb[tile_id] = rgb
                tile_id += 1

            i += 1

    pickle.dump(id_to_rgb, open(OUT_FILE, "wb"))


if __name__ == "__main__":
    parse_tiles()
