# Copyright (c) Facebook, Inc. and its affiliates.
from nle.env import MiniHack
from nle import nethack

COMESTIBLES = [
    "orange",
    "meatball",
    "meat ring",
    "meat stick",
    "kelp frond",
    "eucalyptus leaf",
    "clove of garlic",
    "sprig of wolfsbane",
    "carrot",
    "egg",
    "banana",
    "melon",
    "candy bar",
    "lump of royal jelly",
]

delicious = lambda x: f"This {x} is delicious"
EDIBLE_GOALS = {item: [delicious(item)] for item in COMESTIBLES}
EDIBLE_GOALS.update(
    {
        "apple": ["Delicious!  Must be a Macintosh!", "Core dumped."],
        "pear": ["Core dumped."],
    }
)


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

    def add_object(self, name, symbol="%", loc=None):
        if loc is None:
            loc = "random"
        elif isinstance(loc, tuple) and len(loc) == 2:
            loc = "({},{})".format(loc[0], loc[1])
        elif isinstance(loc, str):
            pass
        else:
            raise ValueError("Invalid location provided.")

        assert isinstance(symbol, str) and len(symbol) == 1
        assert isinstance(name, str)  # TODO maybe check object exists in NetHack

        self.des += "OBJECT:('{}',\"{}\"),{}\n".format(symbol, name, loc)

    def get_des(self):
        return self.des


class MiniHackEat(MiniHack):
    """Environment for "eat" task."""

    def __init__(self, *args, **kwargs):
        kwargs["options"] = kwargs.pop("options", list(nethack.NETHACKOPTIONS))
        kwargs["options"].append("pettype:none")
        kwargs["options"].append("nudist")
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)

        lvl_gen = LevelGenerator(x=8, y=8, lit=True)
        lvl_gen.add_object("apple", "%")

        self.goal_msgs = EDIBLE_GOALS["apple"]

        super().__init__(*args, des_file=lvl_gen.get_des(), **kwargs)

    def _is_episode_end(self, observation):
        """Finish if the message contains the target message. """
        curr_msg = (
            observation[self._original_observation_keys.index("message")]
            .tobytes()
            .decode("utf-8")
        )

        for msg in self.goal_msgs:
            if msg in curr_msg:
                return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING
