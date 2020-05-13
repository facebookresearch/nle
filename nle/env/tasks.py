# Copyright (c) Facebook, Inc. and its affiliates.
import enum

import numpy as np

import nle
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
            penalty. Can be ``constant``, ``exp``, ``square``, ``linear``.
            Defaults to ``constant``.
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
    ) -> None:
        self.penalty_mode = penalty_mode
        self.penalty_step = penalty_step
        self.penalty_time = penalty_time

        self._frozen_steps = 0

        actions = kwargs.pop("actions", TASK_ACTIONS)
        super().__init__(*args, actions=actions, **kwargs)

    def _get_time_penalty(self, last_response, response):
        blstats_old = last_response.Blstats()
        blstats_new = response.Blstats()

        if blstats_old is None or blstats_new is None:
            return 0
        else:
            old_time = blstats_old.Time()
            new_time = blstats_new.Time()

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
            else:  # default
                penalty += self.penalty_step
            penalty += (new_time - old_time) * self.penalty_time
            return penalty

    def _reward_fn(self, last_response, response, end_status):
        """Score delta, but with added a state loop penalty."""
        score_diff = super()._reward_fn(last_response, response, end_status)
        time_penalty = self._get_time_penalty(last_response, response)
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

    def _is_episode_end(self, response) -> None:
        internal = response.Internal()
        if internal and internal.StairsDown():
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_response, response, end_status):
        time_penalty = self._get_time_penalty(last_response, response)
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

    def _is_episode_end(self, response) -> None:
        internal = response.Internal()
        if internal and internal.StairsDown():
            obs = response.Observation()
            s = response.Blstats()
            if obs and s:
                glyphs = (
                    obs.Glyphs()
                    .DataAsNumpy()
                    .view(np.int16)
                    .reshape(base.DUNGEON_SHAPE)
                )
                x = s.CursX()
                y = s.CursY()
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
            if nethack.glyph_to_mon(glyph).mname == "Oracle":
                self.oracle_glyph = glyph
                break
        assert self.oracle_glyph is not None

    def _is_episode_end(self, response) -> None:
        internal = response.Internal()
        if internal:
            obs = response.Observation()
            s = response.Blstats()
            if obs and s:
                glyphs = (
                    obs.Glyphs()
                    .DataAsNumpy()
                    .view(np.int16)
                    .reshape(base.DUNGEON_SHAPE)
                )
                x = s.CursX()
                y = s.CursY()
                neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2].reshape(-1).tolist()
                # TODO: vectorize
                if self.oracle_glyph in neighbors:
                    return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


class NetHackGold(NetHackScore):
    """Environment for the "gold" task.

    The task is similar to the one defined by `NetHackScore`, but the reward
    uses changes in the amount of gold collected by the agent, rather than the
    score.

    The agent will pickup gold automatically by walking on top of it.
    """

    def __init__(
        self, *args, **kwargs,
    ):
        options = kwargs.pop(
            "options",
            (
                "windowtype:rl",
                "color",
                "showexp",
                "nobones",
                "autopickup",
                "pickup_types:$",
            ),
        )

        super().__init__(*args, options=options, **kwargs)

    def _reward_fn(self, last_response, response, end_status):
        """Difference between previous gold and new gold."""
        del end_status  # Unused

        old_gold = base._get(last_response, "Blstats.gold", 0)
        gold = base._get(response, "Blstats.gold", old_gold)
        time_penalty = self._get_time_penalty(last_response, response)

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

    def _reward_fn(self, last_response, response, end_status):
        """Difference between previous hunger and new hunger."""
        del end_status  # Unused

        old_hunger = base._get(last_response, "You.uhunger", 0)
        hunger = base._get(response, "You.uhunger", old_hunger)

        reward = max(0, hunger - old_hunger)

        time_penalty = self._get_time_penalty(last_response, response)

        return reward + time_penalty


class NetHackScout(NetHackScore):
    """Environment for the "scout" task.

    The task is similar to the one defined by `NetHackScore`, but the score is
    defined by the changes in glyphs discovered by the agent.
    """

    def reset(self, *args, **kwargs):
        self.dungeon_explored = {}
        return super().reset(*args, **kwargs)

    def _reward_fn(self, last_response, response, end_status):
        del end_status  # Unused

        reward = 0
        internal = response.Internal()
        if internal:
            obs = response.Observation()
            s = response.Blstats()
            if obs and s:
                glyphs = (
                    obs.Glyphs()
                    .DataAsNumpy()
                    .view(np.int16)
                    .reshape(base.DUNGEON_SHAPE)
                )
                dlevel = response.You().Uz(nle.fbs.DLevel.DLevel())
                dungeon_num = dlevel.Dnum()
                dungeon_level = dlevel.Dlevel()

                key = (dungeon_num, dungeon_level)
                if glyphs is not None:
                    explored = (glyphs != 0).sum()
                    explored_old = 0
                    if key in self.dungeon_explored:
                        explored_old = self.dungeon_explored[key]
                    reward = explored - explored_old
                    self.dungeon_explored[key] = explored
        time_penalty = self._get_time_penalty(last_response, response)
        return reward + time_penalty
