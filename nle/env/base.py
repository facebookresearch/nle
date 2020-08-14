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
import warnings
import weakref

import gym
import numpy as np

from nle import nethack


logger = logging.getLogger(__name__)

WIN_MESSAGE = 1  # Technically dynamic. Practically constant.

# TODO: This doesn't handle all cases, e.g., shopkeepers getting our stuff.
FINAL_QUESTIONS = re.compile(
    rb"Do you want ("
    rb"your possessions identified|"
    rb"to see your attributes|"
    rb"an account of creatures vanquished|"
    rb"to see your conduct|"
    rb"to see the dungeon overview)"
)

DUNGEON_SHAPE = nethack.DUNGEON_SHAPE


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

BLSTATS_SCORE_INDEX = 9


class NLE(gym.Env):
    """Standard NetHack Learning Environment.

    Implements a gym interface around `nethack.Nethack`.


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
            # "killer_name",
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
        observation_keys=(
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
        ),
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
            options (list): list of game options to initialize Nethack. If None,
                Nethack will be initialized with the options found in
                ``nle.nethack.NETHACKOPTIONS`. Defaults to None.
        """

        self.character = character
        self._max_episode_steps = max_episode_steps

        if actions is None:
            actions = FULL_ACTIONS
        self._actions = actions

        self.last_observation = None

        if archivefile is not None:
            warnings.warn("Setting archive file not yet implemented")
            archivefile = None

        if archivefile is not None:
            warnings.warn("Setting archive file not yet implemented")
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

        self._observation_keys = list(observation_keys)

        # Observations we always need.
        for key in (
            "glyphs",
            "blstats",
            "message",
            "program_state",
            "internal",
        ):
            if key not in self._observation_keys:
                self._observation_keys.append(key)

        self._glyph_index = self._observation_keys.index("glyphs")
        self._blstats_index = self._observation_keys.index("blstats")
        self._message_index = self._observation_keys.index("message")
        self._program_state_index = self._observation_keys.index("program_state")
        self._internal_index = self._observation_keys.index("internal")

        self._original_observation_keys = observation_keys
        self._original_indices = tuple(
            self._observation_keys.index(key) for key in observation_keys
        )

        self.env = nethack.Nethack(
            observation_keys=self._observation_keys,
            options=options,
            playername="Agent-" + self.character,
        )
        self._close_env = weakref.finalize(self, lambda e: e.close(), self.env)

        self._random = random.SystemRandom()

        # -1 so that it's 0-based on first reset
        self._episode = -1

        space_dict = {
            "glyphs": gym.spaces.Box(
                low=0, high=nethack.MAX_GLYPH, shape=DUNGEON_SHAPE, dtype=np.int16
            ),
            "chars": gym.spaces.Box(
                low=0, high=255, shape=DUNGEON_SHAPE, dtype=np.uint8
            ),
            "colors": gym.spaces.Box(
                low=0, high=15, shape=DUNGEON_SHAPE, dtype=np.uint8
            ),
            "specials": gym.spaces.Box(
                low=0, high=255, shape=DUNGEON_SHAPE, dtype=np.uint8
            ),
            "blstats": gym.spaces.Box(
                low=np.iinfo(np.int32).min,
                high=np.iinfo(np.int32).max,
                shape=nethack.BLSTATS_SHAPE,
                dtype=np.int32,
            ),
            "message": gym.spaces.Box(
                low=np.iinfo(np.uint8).min,
                high=np.iinfo(np.uint8).max,
                shape=nethack.MESSAGE_SHAPE,
                dtype=np.uint8,
            ),
        }

        self.observation_space = gym.spaces.Dict(
            {key: space_dict[key] for key in observation_keys}
        )

        self.action_space = gym.spaces.Discrete(len(self._actions))

    def _get_observation(self, observation):
        return {
            key: observation[i]
            for key, i in zip(self._original_observation_keys, self._original_indices)
        }

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
        observation, done = self.env.step(self._actions[action])
        observation, done = self._perform_known_steps(observation, done)

        self._steps += 1

        last_observation = self.last_observation
        self.last_observation = observation

        if self._steps >= self._max_episode_steps:
            end_status = self.StepStatus.ABORTED
        else:
            end_status = self._is_episode_end(observation)
        end_status = self.StepStatus(done or end_status)

        reward = float(self._reward_fn(last_observation, observation, end_status))

        if end_status and not done:
            # Try to end the game nicely.
            self._quit_game(observation, done)
            done = True

        info = {}

        if end_status:
            # TODO: fix stats
            # stats = self._collect_stats(last_observation, end_status)
            # stats = stats._asdict()
            stats = {}
            info["stats"] = stats

            if self._stats_logger is not None:
                self._stats_logger.writerow(stats)

        info["end_status"] = end_status

        return self._get_observation(observation), reward, done, info

    def _collect_stats(self, message, end_status):
        """Updates a stats dict tracking several env stats."""
        # Using class rather than instance to allow tasks to reuse this with
        # super()
        # return NLE.Stats(
        #     end_status=int(end_status),
        #     score=_get(message, "Blstats.score", required=True),
        #     time=_get(message, "Blstats.time", required=True),
        #     steps=self._steps,
        #     hp=_get(message, "Blstats.hitpoints", required=True),
        #     exp=_get(message, "Blstats.experience_points", required=True),
        #     exp_lev=_get(message, "Blstats.experience_level", required=True),
        #     gold=_get(message, "Blstats.gold", required=True),
        #     hunger=_get(message, "You.uhunger", required=True),
        #     # killer_name=self._killer_name,
        #     deepest_lev=_get(message, "Internal.deepest_lev_reached", required=True),
        #     episode=self._episode,
        #     seeds=self.get_seeds(),
        #     ttyrec=self.env._process.filename,
        # )

    def _in_moveloop(self, observation):
        program_state = observation[self._program_state_index]
        return program_state[3]  # in_moveloop

    def reset(self):
        """Resets the environment.

        Note:
            We attempt to manually navigate the first few menus so that the
            first seen state is ready to be acted upon by the user. This might
            fail in case Nethack is initialized with some uncommon options.

        Returns:
            (dict): observation of the state as defined by
                    ``self.observation_space``
        """
        self.last_observation = self.env.reset()

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
        # self._killer_name = "UNK"

        self._steps = 0
        self._info = {}

        self._info["seeds"] = {}
        # TODO: Fix seeds
        # for k in nethack.SEED_KEYS:
        #    self._info["seeds"][k] = getattr(self.response.Seeds(), k.capitalize())()

        while not self._in_moveloop(self.last_observation):
            # Get past initial phase of game. This should make sure
            # all the observations are present.
            self.last_observation, done = self.env.step(ASCII_SPACE)
            assert not done, "Game ended unexpectedly"

        return self._get_observation(self.last_observation)

    def close(self):
        self._close_env()
        super().close()

    def get_seeds(self):
        """Returns current seeds.

        Returns:
            (dict): seeds used by the current instance of Nethack.
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
            # TODO: Fix print_message.
            # nhprint.print_message(self.response)
            return
        elif mode == "ansi":
            # TODO(NN): refactor print_message and output string here
            chars_index = self._observation_keys.index("chars")
            chars = self.last_observation[chars_index]
            return "\n".join([line.tobytes().decode("utf-8") for line in chars])
        else:
            return super().render(mode=mode)

    def __repr__(self):
        return "<%s>" % self.__class__.__name__

    def _wait_for_space(self, observation):
        internal = observation[self._internal_index]
        return internal[3]  # xwaitforspace

    def _is_episode_end(self, observation):
        """Returns whether the episode has ended.

        Tasks may override this method to specify different conditions, so long
        as the return value has a well defined __int__ method (e.g. booleans,
        numerical types, enum.IntEnum) and that value is part of StepStatus.

        The return value will be stored into info["end_status"].
        """
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_observation, observation, end_status):
        """Reward function. Difference between previous score and new score."""
        old_score = last_observation[self._blstats_index][BLSTATS_SCORE_INDEX]
        score = observation[self._blstats_index][BLSTATS_SCORE_INDEX]
        del end_status  # Unused for "score" reward.
        return score - old_score

    def _perform_known_steps(self, observation, done, exceptions=True):
        while not done:
            if self._wait_for_space(observation):
                observation, done = self.env.step(ASCII_SPACE)
                continue

            # TODO: Think about killer_name.
            # if self._killer_name == "UNK"

            internal = observation[self._internal_index]
            in_yn_function = internal[1]
            in_getlin = internal[2]

            if in_getlin:  # Game asking for a line of text. We don't do that.
                observation, done = self.env.step(ASCII_ESC)
                continue

            if in_yn_function:  # Game asking for a single character.
                # This causes an annoying unnecessary copy...
                msg = bytes(observation[self._message_index])
                if re.match(FINAL_QUESTIONS, msg):
                    # Auto-yes to the final questions.
                    observation, done = self.env.step(ASCII_y)
                    continue

                if exceptions:
                    # Allow agent to select stuff to eat, attack, and to
                    # select directions.
                    if b"eat" in msg or b"attack" in msg or b"direction?" in msg:
                        break

                # Otherwise, auto-decline.
                observation, done = self.env.step(ASCII_ESC)

            break

        return observation, done

    def _quit_game(self, observation, done):
        """Smoothly quit a game."""
        # Get out of menus and windows.
        observation, done = self._perform_known_steps(
            observation, done, exceptions=False
        )

        if done:
            return

        # Quit the game.
        actions = [0x80 | ord("q"), ord("y")]  # M-q y
        for a in actions:
            observation, done = self.env.step(a)

        # Answer final questions.
        observation, done = self._perform_known_steps(
            observation, done, exceptions=False
        )

        if not done:
            # Somehow, the above logic failed us.
            warnings.warn("Warning: smooth quitting of game failed, aborting.")


def seed_list_to_dict(seeds):
    """Produces seeds dict out of the list of seeds returned by ``env.seed``.

    Arguments:
        seeds (list): list of seeds returned by ``NLE.seed``.

    Returns:
        (dict): seed dictionary correspondent to all seedable env vars.
    """
    return {k: seeds[i] for i, k in enumerate(nethack.SEED_KEYS)}
