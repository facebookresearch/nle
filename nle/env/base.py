# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import csv
import enum
import logging
import os
import random
import re
import sys
import time
import tempfile

import gym
import numpy as np

from nle import nethack
import nle.nethack.print_message as nhprint


logger = logging.getLogger(__name__)

WIN_MESSAGE = 1  # Technically dynamic. Practically constant.

FINAL_QUESTIONS = re.compile(
    rb"Do you want ("
    rb"your possessions identified|"
    rb"to see your attributes|"
    rb"an account of creatures vanquished|"
    rb"to see your conduct|"
    rb"to see the dungeon overview)"
)

DUNGEON_SHAPE = (21, 79)


def _fb_ndarray_to_np(fb_ndarray):
    result = fb_ndarray.DataAsNumpy()
    result = result.view(np.typeDict[fb_ndarray.Dtype()])
    result = result.reshape(fb_ndarray.ShapeAsNumpy().tolist())
    return result


INVFIELDS = [
    "Glyph",
    "Str",
    "Letter",
    "ObjectClass",
    # "ObjectClassName",
]

DEFAULT_MSG_PAD = 256
DEFAULT_INV_PAD = 55
DEFAULT_INVSTR_PAD = 80

ASCII_SPACE = ord(" ")
ASCII_y = ord("y")
ASCII_n = ord("n")
ASCII_ESC = nethack.C("[")

FULL_ACTIONS = list(nethack.ACTIONS)
# Removing some problematic actions
FULL_ACTIONS.remove(nethack.Command.SAVE)
# TODO: consider re-adding help eventually, when we can handle its state machine
# and output.
FULL_ACTIONS.remove(nethack.Command.HELP)
FULL_ACTIONS = tuple(FULL_ACTIONS)


# TODO(NN): this logic should be refactored into a class with pythonic access to
# its attributes.
def _get(response, path="Blstats.score", default=None, required=False):
    node = response

    for attr in path.split("."):
        attr = "".join(x.capitalize() for x in attr.split("_"))
        node = getattr(node, attr)()
        if node is None:
            if required:
                raise ValueError("%s not found in response, but required==True." % path)
            return default
    return node


def _get_glyphs(response):
    if response is None:
        return np.zeros(DUNGEON_SHAPE, dtype=np.int16)
    o = response.Observation()
    # If done is True, Observation() is None.
    if o is None:
        return np.zeros(DUNGEON_SHAPE, dtype=np.int16)
    return o.Glyphs().DataAsNumpy().view(np.int16).reshape(DUNGEON_SHAPE)


def _get_status_fast(response, entries=23):
    # Fast version of _get_status. See that function for the order.
    s = response.Blstats()
    if s is None:
        return np.zeros(entries, dtype=np.int32)
    return np.frombuffer(
        s._tab.Bytes[s._tab.Pos : s._tab.Pos + 4 * entries], dtype=np.int32
    )


def _get_status(response):
    s = response.Blstats()
    if s is None:
        return np.zeros(23, dtype=np.int32)

    return np.array(
        (
            s.CursX(),
            s.CursY(),
            # 1..125. See
            # https://nethackwiki.com/wiki/Attribute#Strength_in_game_formulas
            # and get_strength_str() in botl.c
            s.StrengthPercentage(),
            s.Strength(),
            s.Dexterity(),
            s.Constitution(),
            s.Intelligence(),
            s.Wisdom(),
            s.Charisma(),
            s.Score(),
            s.Hitpoints(),
            s.MaxHitpoints(),
            s.Depth(),
            s.Gold(),
            s.Energy(),
            s.MaxEnergy(),
            s.ArmorClass(),
            s.MonsterLevel(),
            s.ExperienceLevel(),
            s.ExperiencePoints(),
            s.Time(),
            s.HungerState(),
            s.CarryingCapacity(),
        ),
        dtype=np.int32,
    )


def _get_padded_message(
    response, padded_length=DEFAULT_MSG_PAD, alphabet_size=ord("~") - ord(" ") + 1
):
    # TODO: This would be faster if done in C++.
    result = np.full(padded_length, fill_value=alphabet_size, dtype=np.uint8)
    if response is None or response.NotRunning():
        return result

    win = response.Windows(WIN_MESSAGE)
    if win is None:
        logger.error("No message window. This shouldn't happen.")
        return result

    assert win.Type() == nethack.NHW_MESSAGE

    offset = 0
    for i in range(win.StringsLength()):
        message = np.frombuffer(win.Strings(i), dtype=np.uint8)

        # Subtract ord(" ") and assign. Crop if space runs out.
        result[offset : offset + len(message)] = (
            message[: max(len(result) - offset, 0)] - 0x20
        )
        offset += len(message) + 1  # Keep one separation token.
    return result


def _wait_for_space(response):
    internal = response.Internal()
    return internal and internal.Xwaitforspace()


def _in_moveloop(response):
    ps = response.ProgramState()
    return ps and ps.InMoveloop()


def _get_call_stack(response):
    internal = response.Internal()
    if internal is None:
        return ()
    return (internal.CallStack(i) for i in range(internal.CallStackLength()))


def _get_inv(response):
    result = {}
    for field in INVFIELDS:
        result[field] = []

    o = response.Observation()
    if o is None:
        return result

    for item in (o.Inventory(i) for i in range(o.InventoryLength())):
        for field in INVFIELDS:
            result[field].append(getattr(item, field)())
    return result


def _get_padded_inv(
    response,
    padded_length=DEFAULT_INV_PAD,
    str_padded_length=DEFAULT_INVSTR_PAD,
    alphabet_size=ord("~") - ord(" ") + 1,
):
    # TODO: This would be faster if done in C++.
    inv = _get_inv(response)
    strs = np.full(
        (padded_length, str_padded_length), fill_value=alphabet_size, dtype=np.uint8
    )
    for i, b in enumerate(inv["Str"]):
        strs[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)[:str_padded_length] - 0x20

    pad_width = (0, padded_length - len(inv["Str"]))
    glyphs = np.pad(
        np.asarray(inv["Glyph"], dtype=np.int16),
        pad_width,
        mode="constant",
        constant_values=nethack.NO_GLYPH,
    )
    letters = np.pad(
        np.asarray(inv["Letter"], dtype=np.uint8) - 0x20,
        pad_width,
        mode="constant",
        constant_values=alphabet_size,
    )
    oclasses = np.pad(
        np.asarray(inv["ObjectClass"], dtype=np.uint8),
        pad_width,
        mode="constant",
        constant_values=nethack.MAXOCLASSES,
    )

    return glyphs, strs, letters, oclasses


class NLE(gym.Env):
    """Standard NetHack Learning Environment.

    Implements a gym interface around `nethack.NetHack`.


    Examples:
        >>> env = NLE()
        >>> obs = env.reset()
        >>> obs, reward, done, info = env.step(0)
        >>> env.render()
    """

    metadata = {"render.modes": ["human", "ansi"]}

    class StepStatus(enum.IntEnum):
        """Specifies the status of the terminal state.

        Note:
            User may redefine this class in subtasks to handle / categorize
            more terminal states.

            It is highly advised that, in such cases, the enums already defined
            in this object are replicated in some way. See `nle.env.tasks` for
            examples on how to do this right.
        """

        ABORTED = -1
        RUNNING = 0
        DEATH = 1

    Stats = collections.namedtuple(
        "Stats",
        (
            "end_status",
            "score",
            "time",
            "steps",
            "hp",
            "exp",
            "exp_lev",
            "gold",
            "hunger",
            "killer_name",
            "deepest_lev",
            "episode",
            "seeds",
            "ttyrec",
        ),
    )

    def __init__(
        self,
        savedir=None,
        archivefile="nethack.%(pid)i.%(time)s.zip",
        character="mon-hum-neu-mal",
        max_episode_steps=5000,
        observation_keys=("glyphs", "status", "message", "inventory"),
        actions=None,
        options=None,
    ):
        """Constructs a new NLE environment.

        Args:
            savedir (str or None): path to save archives into. Defaults to None.
            archivefile (str or None): Template for the zip archive filename of
                NetHack ttyrec files. Use "%(pid)i" for the process id of the
                NetHack process, "%(time)s" for the creation time. Use None to
                disable writing archivefiles.
            character (str): name of character. Defaults to "mon-hum-neu-mal".
            max_episode_steps (int): maximum amount of steps allowed before the
                game is forcefully quit. In such cases, ``info["end_status"]``
                will be equal to ``StepStatus.ABORTED``. Defaults to 5000.
            observation_keys (list): keys to use when creating the observation.
                Defaults to all.
            actions (list): list of actions. If None, the full action space will
                be used, i.e. ``nle.nethack.ACTIONS``. Defaults to None.
            options (list): list of game options to initialize NetHack. If None,
                NetHack will be initialized with the options found in
                ``nle.nethack.NETHACKOPTIONS`. Defaults to None.
        """

        self.character = character
        self._max_episode_steps = max_episode_steps

        if actions is None:
            actions = FULL_ACTIONS
        self._actions = actions

        self.response = None

        if archivefile is not None:
            try:
                if savedir is None:
                    parent_dir = os.path.join(os.getcwd(), "nle_data")
                    os.makedirs(parent_dir, exist_ok=True)
                    # Create a unique subdirectory for us.
                    self.savedir = tempfile.mkdtemp(
                        prefix=time.strftime("%Y%m%d-%H%M%S_"), dir=parent_dir
                    )
                else:
                    self.savedir = savedir
                    os.makedirs(self.savedir)
            except FileExistsError:
                logger.info("Using existing savedir: %s", self.savedir)
            else:
                logger.info("Created savedir: %s", self.savedir)

            self.archivefile = os.path.join(self.savedir, archivefile)
        else:
            self.savedir = None
            self.archivefile = None
            self._stats_file = None
            self._stats_logger = None

        self._setup_statsfile = archivefile is not None

        self.env = nethack.NetHack(
            archivefile=self.archivefile,
            options=options,
            playername="Agent%(pid)i-" + self.character,
        )

        self._random = random.SystemRandom()

        # -1 so that it's 0-based on first reset
        self._episode = -1

        space_dict = {
            "glyphs": gym.spaces.Box(
                low=np.iinfo(np.int16).min,
                high=np.iinfo(np.int16).max,
                shape=DUNGEON_SHAPE,
                dtype=np.int16,
            ),
            "status": gym.spaces.Box(
                low=np.iinfo(np.int32).min,
                high=np.iinfo(np.int32).max,
                shape=(23,),
                dtype=np.int32,
            ),
            "message": gym.spaces.Box(
                low=np.iinfo(np.uint8).min,
                high=np.iinfo(np.uint8).max,
                shape=(DEFAULT_MSG_PAD,),
                dtype=np.uint8,
            ),
            "inventory": gym.spaces.Tuple(
                (
                    gym.spaces.Box(
                        low=np.iinfo(np.int16).min,
                        high=np.iinfo(np.int16).max,
                        shape=(DEFAULT_INV_PAD,),
                        dtype=np.int16,
                    ),
                    gym.spaces.Box(
                        low=np.iinfo(np.uint8).min,
                        high=np.iinfo(np.uint8).max,
                        shape=(DEFAULT_INV_PAD, DEFAULT_INVSTR_PAD),
                        dtype=np.uint8,
                    ),
                    gym.spaces.Box(
                        low=np.iinfo(np.uint8).min,
                        high=np.iinfo(np.uint8).max,
                        shape=(DEFAULT_INV_PAD,),
                        dtype=np.uint8,
                    ),
                    gym.spaces.Box(
                        low=np.iinfo(np.uint8).min,
                        high=np.iinfo(np.uint8).max,
                        shape=(DEFAULT_INV_PAD,),
                        dtype=np.uint8,
                    ),
                )
            ),
        }

        self.observation_space = gym.spaces.Dict(
            {key: space_dict[key] for key in observation_keys}
        )

        self._key_functions = {
            "glyphs": _get_glyphs,
            "status": _get_status_fast,
            "message": _get_padded_message,
            "inventory": _get_padded_inv,
        }
        for key in list(self._key_functions.keys()):
            if key not in observation_keys:
                del self._key_functions[key]

        self.action_space = gym.spaces.Discrete(len(self._actions))

    def _get_observation(self, response):
        return {key: f(response) for key, f in self._key_functions.items()}

    def step(self, action: int):
        """Steps the environment.

        Args:
            action (int): action integer as defined by ``self.action_space``.

        Returns:
            (dict, float, bool, dict): a tuple containing
                - (*dict*): an observation of the state; this will contain the keys
                  specified by ``self.observation_space``.
                - (*float*): a reward; see ``self._reward_fn`` to see how it is
                  specified.
                - (*bool*): True if the state is terminal, False otherwise.
                - (*dict*): a dictionary of extra information (such as
                  `end_status`, i.e. a status info -- death, task win, etc. --
                  for the terminal state).
        """
        response, done, info = self.env.step(self._actions[action])
        response, done, info = self._perform_known_steps(response, done, info)

        self._steps += 1

        last_response = self.response
        self.response = response

        if self._steps >= self._max_episode_steps:
            end_status = self.StepStatus.ABORTED
        else:
            end_status = self._is_episode_end(response)
        end_status = self.StepStatus(done or end_status)

        reward = float(self._reward_fn(last_response, response, end_status))

        if end_status and not done:
            # Try to end the game nicely.
            self._quit_game(response, done, info)
            done = True

        if end_status:
            stats = self._collect_stats(last_response, end_status)
            stats = stats._asdict()
            info["stats"] = stats

            if self._stats_logger is not None:
                self._stats_logger.writerow(stats)

        info["end_status"] = end_status

        return self._get_observation(response), reward, done, info

    def _collect_stats(self, message, end_status):
        """Updates a stats dict tracking several env stats."""
        # Using class rather than instance to allow tasks to reuse this with
        # super()
        return NLE.Stats(
            end_status=int(end_status),
            score=_get(message, "Blstats.score", required=True),
            time=_get(message, "Blstats.time", required=True),
            steps=self._steps,
            hp=_get(message, "Blstats.hitpoints", required=True),
            exp=_get(message, "Blstats.experience_points", required=True),
            exp_lev=_get(message, "Blstats.experience_level", required=True),
            gold=_get(message, "Blstats.gold", required=True),
            hunger=_get(message, "You.uhunger", required=True),
            killer_name=self._killer_name,
            deepest_lev=_get(message, "Internal.deepest_lev_reached", required=True),
            episode=self._episode,
            seeds=self.get_seeds(),
            ttyrec=self.env._process.filename,
        )

    def reset(self):
        """Resets the environment.

        Note:
            We attempt to manually navigate the first few menus so that the
            first seen state is ready to be acted upon by the user. This might
            fail in case NetHack is initialized with some uncommon options.

        Returns:
            (dict): observation of the state as defined by
                    ``self.observation_space``
        """
        self.response = self.env.reset()

        # Only run on the first reset to initialize stats file
        if self._setup_statsfile:
            stats_file = os.path.splitext(self.env._archive.filename)[0] + ".csv"
            add_header = not os.path.exists(stats_file)

            self._stats_file = open(stats_file, "a", 1)  # line buffered
            self._stats_logger = csv.DictWriter(
                self._stats_file, fieldnames=self.Stats._fields
            )
            if add_header:
                self._stats_logger.writeheader()
        self._setup_statsfile = False

        self._episode += 1
        self._killer_name = "UNK"

        self._steps = 0
        self._info = self.env._info.copy()

        self._info["seeds"] = {}
        for k in nethack.SEED_KEYS:
            self._info["seeds"][k] = getattr(self.response.Seeds(), k.capitalize())()

        while not _in_moveloop(self.response):
            # Get past initial phase of game. This should make sure
            # all the observations are present.
            self.response, done, _ = self.env.step(ASCII_SPACE)
            assert not done, "Game ended unexpectedly"

        return self._get_observation(self.response)

    def get_seeds(self):
        """Returns current seeds.

        Returns:
            (dict): seeds used by the current instance of NetHack.
        """

        return self._info["seeds"].copy()

    def seed(self, seeds=None):
        """Seeds the environment.

        Won't take effect until the environment is reset. The game is seeded
        using the provided seed(s) each time that the environment is reset. Thus
        the environment will return one and only one instance of NetHack per
        seed.

        Arguments:
            seeds (dict): seed used on NetHack process upon reset. If type is
                          ``int``, a list of seeds gets generated by
                          incrementing the provided number. If ``None``, seeds
                          are generayed by NLE using ``self._random``.

        Returns:
            (list): the seeds used by NetHack (in a list to respect the API
                    defined by ``gym``). Use ``nle.seed_list_to_dict`` to get the
                    respective dictionary.
        """
        if seeds is None:
            seeds = {}
            for k in nethack.SEED_KEYS:
                seeds[k] = self._random.randrange(sys.maxsize)
        elif isinstance(seeds, int):
            to_add = 0
            seed_dict = {}
            for k in nethack.SEED_KEYS:
                to_add += 1
                seed_dict[k] = seeds + to_add
            seeds = seed_dict

        self.env.seed(seeds)
        return list(seeds.values())

    def render(self, mode="human"):
        """Renders the state of environment.

        Arguments:
           mode (str): Defaults to "human". Acceptable values are "human" and
                       "ansi", otherwise a ``ValueError`` is raised.

        """
        if mode == "human":
            nhprint.print_message(self.response)
            return
        elif mode == "ansi":
            # TODO(NN): refactor print_message and output string here
            chars = _get(self.response, "Observation.Chars")
            if chars is None:
                return ""
            chars = _fb_ndarray_to_np(chars)
            return "\n".join([line.tobytes().decode("utf-8") for line in chars])
        else:
            return super().render(mode=mode)

    def __repr__(self):
        return "<%s>" % self.__class__.__name__

    def _is_episode_end(self, response):
        """Returns whether the episode has ended.

        Tasks may override this method to specify different conditions, so long
        as the return value has a well defined __int__ method (e.g. booleans,
        numerical types, enum.IntEnum) and that value is part of StepStatus.

        The return value will be stored into info["end_status"].
        """
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_response, response, end_status):
        """Reward function. Difference between previous score and new score."""
        old_score = _get(last_response, "Blstats.score", 0)
        score = _get(response, "Blstats.score", old_score)
        del end_status  # Unused for "score" reward.
        return score - old_score

    def _perform_known_steps(self, response, done, info):
        while not done:
            if _wait_for_space(response):
                response, done, info = self.env.step(ASCII_SPACE)
                continue

            if self._killer_name == "UNK" and response.ProgramState().Gameover():
                self._killer_name = _get(
                    response, "Internal.killer_name", default="UNK"
                )
                if self._killer_name != "UNK":
                    self._killer_name = self._killer_name.decode("utf-8")

            message_win = response.Windows(WIN_MESSAGE)
            if message_win is None or not message_win.StringsLength():
                break
            msg = message_win.Strings(message_win.StringsLength() - 1)

            if msg.startswith(b"Beware, there will be no return!  Still climb?"):
                response, done, info = self.env.step(ASCII_n)
            elif re.match(FINAL_QUESTIONS, msg):
                response, done, info = self.env.step(ASCII_y)
            else:
                call_stack = _get_call_stack(response)
                if b"yn_function" in call_stack or b"getlin" in call_stack:
                    if b"eat" in msg or b"attack" in msg or b"direction?" in msg:
                        break
                    response, done, info = self.env.step(ASCII_ESC)
                else:
                    break

        return response, done, info

    def _quit_game(self, response, done, info):
        """Smoothly quit a game."""
        # Get out of menus and windows.
        response, done, info = self._perform_known_steps(response, done, info)

        if done:
            return

        # Quit the game.
        actions = "#quit\ny"
        for a in actions:
            response, done, info = self.env.step(ord(a))

        # Answer final questions.
        response, done, info = self._perform_known_steps(response, done, info)

        if not done:
            # Somehow, the above logic failed us. We'll SIGTERM the game to close it.
            if self.env._archive is None:
                filename = "N/A"
            else:
                filename = self.env._archive.filename
            logger.error(
                "Warning: smooth quitting of game failed, aborting "
                "(archive %s, episode %i).",
                filename,
                self.env._episode,
            )


def seed_list_to_dict(seeds):
    """Produces seeds dict out of the list of seeds returned by ``env.seed``.

    Arguments:
        seeds (list): list of seeds returned by ``NLE.seed``.

    Returns:
        (dict): seed dictionary correspondent to all seedable env vars.
    """
    return {k: seeds[i] for i, k in enumerate(nethack.SEED_KEYS)}
