# Copyright (c) Facebook, Inc. and its affiliates.
import enum

import numpy as np

from nle.env import base
from nle import nethack


TASK_ACTIONS = tuple(
    [nethack.MiscAction.MORE]
    + list(nethack.CompassDirection)
    + list(nethack.CompassDirectionLonger)
    + list(nethack.MiscDirection)
    + [nethack.Command.KICK, nethack.Command.EAT, nethack.Command.SEARCH]
)


class NetHackScore(base.NLE):
    """Environment for "score" task.

    The task is an augmentation of the standard NLE task. The return function is
    defined as:
    :math:`\text{score}_t - \text{score}_{t-1} + \text{TP}`,
    where the :math:`\text{TP}` is a time penalty that grows with the amount of
    environment steps that do not change the state (such as navigating menus).

    Args:
        penalty_mode (str): name of the mode for calculating the time step
            penalty. Can be ``constant``, ``exp``, ``square``, ``linear``, or
            ``always``. Defaults to ``constant``.
        penalty_step (float): constant applied to amount of frozen steps.
            Defaults to -0.01.
        penalty_time (float): constant applied to amount of frozen steps.
            Defaults to -0.0.

    """

    def __init__(
        self,
        *args,
        penalty_mode="constant",
        penalty_step: float = -0.01,
        penalty_time: float = -0.0,
        **kwargs,
    ):
        self.penalty_mode = penalty_mode
        self.penalty_step = penalty_step
        self.penalty_time = penalty_time

        self._frozen_steps = 0

        actions = kwargs.pop("actions", TASK_ACTIONS)
        super().__init__(*args, actions=actions, **kwargs)

    def _get_time_penalty(self, last_observation, observation):
        blstats_old = last_observation[self._blstats_index]
        blstats_new = observation[self._blstats_index]

        old_time = blstats_old[20]  # moves
        new_time = blstats_new[20]  # moves

        if old_time == new_time:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0

        penalty = 0
        if self.penalty_mode == "constant":
            if self._frozen_steps > 0:
                penalty += self.penalty_step
        elif self.penalty_mode == "exp":
            penalty += 2 ** self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "square":
            penalty += self._frozen_steps ** 2 * self.penalty_step
        elif self.penalty_mode == "linear":
            penalty += self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "always":
            penalty += self.penalty_step
        else:  # default
            raise ValueError("Unknown penalty_mode '%s'" % self.penalty_mode)
        penalty += (new_time - old_time) * self.penalty_time
        return penalty

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Score delta, but with added a state loop penalty."""
        score_diff = super()._reward_fn(
            last_observation, action, observation, end_status
        )
        time_penalty = self._get_time_penalty(last_observation, observation)
        return score_diff + time_penalty


class NetHackStaircase(NetHackScore):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.
    """

    class StepStatus(enum.IntEnum):
        ABORTED = -1
        RUNNING = 0
        DEATH = 1
        TASK_SUCCESSFUL = 2

    def _is_episode_end(self, observation):
        internal = observation[self._internal_index]
        stairs_down = internal[4]
        if stairs_down:
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_observation, action, observation, end_status):
        del action  # Unused
        time_penalty = self._get_time_penalty(last_observation, observation)
        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward = 1
        else:
            reward = 0
        return reward + time_penalty


class NetHackStaircasePet(NetHackStaircase):
    """Environment for "staircase-pet" task.

    This task requires the agent to get on top of a staircase down (>), while
    having their pet next to it. See `NetHackStaircase` for the reward function.
    """

    def _is_episode_end(self, observation):
        internal = observation[self._internal_index]
        stairs_down = internal[4]
        if stairs_down:
            glyphs = observation[self._glyph_index]
            blstats = observation[self._blstats_index]
            x, y = blstats[:2]

            neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2].reshape(-1).tolist()
            # TODO: vectorize
            for glyph in neighbors:
                if nethack.glyph_is_pet(glyph):
                    return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


class NetHackOracle(NetHackStaircase):
    """Environment for "oracle" task.

    This task requires the agent to reach the oracle (by standing next to it).
    See `NetHackStaircase` for the reward function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_glyph = None
        for glyph in range(nethack.GLYPH_MON_OFF, nethack.GLYPH_PET_OFF):
            if nethack.permonst(nethack.glyph_to_mon(glyph)).mname == "Oracle":
                self.oracle_glyph = glyph
                break
        assert self.oracle_glyph is not None

    def _is_episode_end(self, observation):
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]
        x, y = blstats[:2]

        neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]
        if np.any(neighbors == self.oracle_glyph):
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


class NetHackGold(NetHackScore):
    """Environment for the "gold" task.

    The task is similar to the one defined by `NetHackScore`, but the reward
    uses changes in the amount of gold collected by the agent, rather than the
    score.

    The agent will pickup gold automatically by walking on top of it.
    """

    def __init__(self, *args, **kwargs):
        options = kwargs.pop("options", None)

        if options is None:
            # Copy & swap out "pickup_types".
            options = []
            for option in nethack.NETHACKOPTIONS:
                if option.startswith("pickup_types"):
                    options.append("pickup_types:$")
                    continue
                options.append(option)

        super().__init__(*args, options=options, **kwargs)

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Difference between previous gold and new gold."""
        del end_status  # Unused
        del action  # Unused
        if not self.env.in_normal_game():
            # Before game started or after it ended stats are zero.
            return 0.0

        old_blstats = last_observation[self._blstats_index]
        blstats = observation[self._blstats_index]

        old_gold = old_blstats[13]
        gold = blstats[13]

        time_penalty = self._get_time_penalty(last_observation, observation)

        return gold - old_gold + time_penalty


# FIXME: the way the reward function is currently structured means the
# agents gets a penalty of -1 every other step (since the
# uhunger increases by that)
# thus the step penalty becomes irrelevant
class NetHackEat(NetHackScore):
    """Environment for the "eat" task.

    The task is similar to the one defined by `NetHackScore`, but the reward
    uses positive changes in the character's hunger level (e.g. by consuming
    comestibles or monster corpses), rather than the score.
    """

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Difference between previous hunger and new hunger."""
        del end_status  # Unused
        del action  # Unused

        if not self.env.in_normal_game():
            # Before game started or after it ended stats are zero.
            return 0.0

        old_internal = last_observation[self._internal_index]
        internal = observation[self._internal_index]

        old_uhunger = old_internal[7]
        uhunger = internal[7]

        reward = max(0, uhunger - old_uhunger)

        time_penalty = self._get_time_penalty(last_observation, observation)

        return reward + time_penalty


class NetHackScout(NetHackScore):
    """Environment for the "scout" task.

    The task is similar to the one defined by `NetHackScore`, but the score is
    defined by the changes in glyphs discovered by the agent.
    """

    def reset(self, *args, **kwargs):
        self.dungeon_explored = {}
        return super().reset(*args, **kwargs)

    def _reward_fn(self, last_observation, action, observation, end_status):
        del end_status  # Unused
        del action  # Unused

        if not self.env.in_normal_game():
            # Before game started or after it ended stats are zero.
            return 0.0

        reward = 0
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]

        dungeon_num, dungeon_level = blstats[23:25]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != 0)
        explored_old = 0
        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = explored - explored_old
        self.dungeon_explored[key] = explored
        time_penalty = self._get_time_penalty(last_observation, observation)
        return reward + time_penalty


class NetHackChallenge(NetHackScore):
    """Environment for the NetHack Challenge.

    The task is an augmentation of the standard NLE task. This is the NLE Score Task
    but with some subtle differences:
        * the action space is fixed to include the full keyboard
        * menus and "<More>" tokens are not skipped
        * starting character is randomly assigned
    """

    def __init__(
        self,
        *args,
        character="@",
        allow_all_yn_questions=True,
        allow_all_modes=True,
        penalty_mode="constant",
        penalty_step: float = -0.00,
        penalty_time: float = -0.0,
        max_episode_steps: int = 1e6,
        observation_keys=(
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
            "inv_glyphs",
            "inv_strs",
            "inv_letters",
            "inv_oclasses",
            "tty_chars",
            "tty_colors",
            "tty_cursor",
            "misc",
        ),
        no_progress_timeout: int = 10_000,
        **kwargs,
    ):
        actions = nethack.ACTIONS
        super().__init__(
            *args,
            actions=actions,
            character=character,
            allow_all_yn_questions=allow_all_yn_questions,
            allow_all_modes=allow_all_modes,
            penalty_mode=penalty_mode,
            penalty_step=penalty_step,
            penalty_time=penalty_time,
            max_episode_steps=max_episode_steps,
            observation_keys=observation_keys,
            **kwargs,
        )
        # If the in-game turn count doesn't change for 10_000 steps, we abort
        self.no_progress_timeout = no_progress_timeout

        def f(*args, **kwargs):
            raise RuntimeError("Should not try changing seeds")

        self.env.set_initial_seeds = f
        self.env.set_current_seeds = f
        self.env.get_current_seeds = f

    def reset(self, *args, **kwargs):
        self._turns = None
        self._no_progress_count = 0
        return super().reset(*args, **kwargs)

    def _check_abort(self, observation):
        """Check if time has stopped and no observations has changed long enough
        to trigger an abort."""

        turns = observation[self._blstats_index][20]
        if self._turns == turns:
            self._no_progress_count += 1
        else:
            self._turns = turns
            self._no_progress_count = 0
        return (
            self._steps >= self._max_episode_steps
            or self._no_progress_count >= self.no_progress_timeout
        )

    def seed(self, core=None, disp=None, reseed=True):
        raise RuntimeError("NetHackChallenge doesn't allow seed changes")
