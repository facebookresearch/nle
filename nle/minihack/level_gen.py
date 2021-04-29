# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np

MAZE_FLAGS = (
    "noteleport",
    "hardfloor",
    "nommap",
    "arboreal",
    "shortsighted",
    "mazelevel",
    "premapped",
    "shroud",
    "graveyard",
    "icedpools",
    "solidify",
    "corrmaze",
    "inaccessibles",
)


class LevelGenerator:
    def __init__(
        self,
        map=None,
        w=8,
        h=8,
        lit=True,
        message="Welcome to MiniHack!",
        flags=("noteleport", "hardfloor"),
    ):
        assert all(
            f in MAZE_FLAGS for f in flags
        ), "One of the provided maze flags is incorrect"
        flags_str = ",".join(flags)

        self.header = f"""
MAZE: "mylevel", ' '
FLAGS:{flags_str}
MESSAGE: \"{message}\"
GEOMETRY:center,center
"""

        self.mapify = lambda x: "MAP\n" + x + "ENDMAP\n"
        self.init_map(map, w, h)

        litness = "lit" if lit else "unlit"
        self.footer = f'REGION:(0,0,{self.x},{self.y}),{litness},"ordinary"\n'

        self.stair_up_exist = False

    def init_map(self, map=None, x=8, y=8):
        if map is None:
            # Creating empty area
            self.x = x
            self.y = y
            self.map = np.array([["."] * x] * y, dtype=str)
        else:
            lines = [list(line) for line in map.split("\n") if len(line) > 0]
            self.y = len(lines)
            self.x = max(len(line) for line in lines)
            new_lines = [line + [" "] * (self.x - len(line)) for line in lines]
            self.map = np.array(new_lines)

    def get_map_str(self):
        """Returns the map as a string."""
        map_list = ["".join(self.map[i]) + "\n" for i in range(self.map.shape[0])]
        return "".join(map_list)

    def get_map_array(self):
        """Returns the map as as np array."""
        return self.map

    def get_des(self):
        """Returns the description file."""
        return self.header + self.mapify(self.get_map_str()) + self.footer

    @staticmethod
    def validate_place(place):
        if place is None:
            place = "random"
        elif isinstance(place, tuple):
            place = LevelGenerator.validate_coord(place)
            place = str(place)
        elif isinstance(place, str):
            pass
        else:
            raise ValueError("Invalid place provided.")

        return place

    @staticmethod
    def validate_coord(coord):
        assert (
            isinstance(coord, tuple)
            and len(coord) == 2
            and isinstance(coord[0], int)
            and isinstance(coord[1], int)
        )
        return coord

    def add_object(self, name, symbol="%", place=None, cursestate=None):
        place = self.validate_place(place)
        assert isinstance(symbol, str) and len(symbol) == 1
        assert isinstance(name, str)  # TODO maybe check object exists in NetHack

        self.footer += f"OBJECT:('{symbol}',\"{name}\"), {place}"

        if cursestate is not None:
            assert cursestate in ["blessed", "uncursed", "cursed", "random"]
            if cursestate != "random":
                self.footer += f", {cursestate}"

        self.footer += "\n"

    def add_terrain(self, coord, flag, in_footer=False):
        coord = self.validate_coord(coord)

        if in_footer:
            assert flag in ["-", "F", "L", "T", "C"]
            self.footer += f"TERRAIN: {str(coord)}, '{flag}'\n"
        else:
            assert flag in [".", " ", "-", "F", "L", "T", "C", "}"]
            x, y = coord
            self.map[y, x] = flag

    def add_stair_down(self, place=None):
        place = self.validate_place(place)
        self.footer += f"STAIR:{place},down\n"

    def add_stair_up(self, coord):
        if self.stair_up_exist:
            return
        x, y = self.validate_coord(coord)
        _x, _y = abs(x - 1), abs(y - 1)
        self.footer += f"BRANCH:({x},{y},{x},{y}),({_x},{_y},{_x},{_y})\n"
        self.stair_up_exist = True

    def add_altar(self, place=None):
        place = self.validate_place(place)
        self.footer += f"ALTAR:{place},neutral,altar\n"

    def add_sink(self, place=None):
        place = self.validate_place(place)
        self.footer += f"SINK:{place}\n"
