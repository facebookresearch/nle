from collections import defaultdict
from nle.env import tasks
import numpy as np


class SharedPatch(object):
    def __init__(self, *args, state_counter="none", **kwargs):
        # intialize state counter
        self.state_counter = state_counter
        if self.state_counter != "none":
            self.state_count_dict = defaultdict(int)
        # this super() goes to the parent of the particular task, not to `object`
        super().__init__(*args, **kwargs)

    def step(self, action):
        # add state counting to step function if desired
        step_return = super().step(action)
        if self.state_counter == "none":
            # do nothing
            return step_return

        obs, reward, done, info = step_return

        if self.state_counter == "ones":
            # treat every state as unique
            state_visits = 1
        elif self.state_counter == "coordinates":
            # use the location of the agent within the dungeon to accumulate visits
            features = obs["blstats"]
            x = features[0]
            y = features[1]
            # TODO: prefer to use dungeon level and dungeon number from Blstats
            d = features[12]
            coord = (d, x, y)
            self.state_count_dict[coord] += 1
            state_visits = self.state_count_dict[coord]
        else:
            raise NotImplementedError("state_counter=%s" % self.state_counter)

        obs.update(state_visits=np.array([state_visits]))

        if done:
            self.state_count_dict.clear()

        return step_return

    def reset(self, wizkit_items=None):
        # reset state counter when env resets
        obs = super().reset(wizkit_items=wizkit_items)
        if self.state_counter != "none":
            self.state_count_dict.clear()
            # current state counts as one visit
            obs.update(state_visits=np.array([1]))
        return obs


class PatchedNetHackScore(SharedPatch, tasks.NetHackScore):
    pass


class PatchedNetHackStaircase(SharedPatch, tasks.NetHackStaircase):
    def __init__(self, *args, reward_win=1, reward_lose=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_win = reward_win
        self.reward_lose = reward_lose

    def _reward_fn(self, last_response, response, end_status):
        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward = self.reward_win
        elif end_status == self.StepStatus.RUNNING:
            reward = 0
        else:  # death or aborted
            reward = self.reward_lose
        return reward + self._get_time_penalty(last_response, response)


class PatchedNetHackStaircasePet(PatchedNetHackStaircase, tasks.NetHackStaircasePet):
    pass  # inherit from PatchedNetHackStaircase


class PatchedNetHackStaircaseOracle(PatchedNetHackStaircase, tasks.NetHackOracle):
    pass  # inherit from PatchedNetHackStaircase


class PatchedNetHackGold(SharedPatch, tasks.NetHackGold):
    pass


class PatchedNetHackEat(SharedPatch, tasks.NetHackEat):
    pass


class PatchedNetHackScout(SharedPatch, tasks.NetHackScout):
    pass


NetHackScore = PatchedNetHackScore
NetHackStaircase = PatchedNetHackStaircase
NetHackStaircasePet = PatchedNetHackStaircasePet
NetHackOracle = PatchedNetHackStaircaseOracle
NetHackGold = PatchedNetHackGold
NetHackEat = PatchedNetHackEat
NetHackScout = PatchedNetHackScout


ENVS = dict(
    staircase=NetHackStaircase,
    score=NetHackScore,
    pet=NetHackStaircasePet,
    oracle=NetHackOracle,
    gold=NetHackGold,
    eat=NetHackEat,
    scout=NetHackScout,
)
