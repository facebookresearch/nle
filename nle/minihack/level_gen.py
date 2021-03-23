class LevelGenerator:
    def __init__(self, map=None, x=8, y=8, lit=True):

        self.header = """
MAZE: "mylevel", ' '
INIT_MAP:solidfill,' '
GEOMETRY:center,center
"""
        # TODO add more flag options?
        self.des = self.header

        mapify = lambda x: "MAP\n" + x + "ENDMAP\n"
        if map is not None:
            self.des += mapify(map)
            self.x = map.count("\n")
            self.y = max([len(line) for line in map.split("\n")])
        else:
            self.x = x
            self.y = y
            # Creating empty area
            row = "." * y + "\n"
            maze = row * x
            self.des += mapify(maze)

        litness = "lit" if lit else "unlit"
        self.des += 'REGION:(0,0,{},{}),{},"ordinary"\n'.format(x, y, litness)

        self.stair_up_exist = False

    def get_des(self):
        return self.des

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

        self.des += f"OBJECT:('{symbol}',\"{name}\"), {place}"

        if cursestate is not None:
            assert cursestate in ["blessed", "uncursed", "cursed", "random"]
            if cursestate != "random":
                self.des += f", {cursestate}"

        self.des += "\n"

    def add_terrain(self, coord, flag):
        coord = str(self.validate_coord(coord))
        assert flag in ["-", "F", "L", "T", "C"]

        self.des += f"TERRAIN: {coord}, '{flag}'\n"

    def add_stair_down(self, place=None):
        place = self.validate_place(place)
        self.des += f"STAIR:{place},down\n"

    def add_stair_up(self, coord):
        if self.stair_up_exist:
            return
        x, y = self.validate_coord(coord)
        self.des += f"BRANCH:{x, y, x, y},(0,0,0,0)\n"
        self.stair_up_exist = True

    def add_altar(self, place=None):
        place = self.validate_place(place)
        self.des += f"ALTAR:{place},neutral,altar\n"

    def add_sink(self, place=None):
        place = self.validate_place(place)
        self.des += f"SINK:{place}\n"
