from collections import defaultdict
import gym
import numpy as np


class CounterWrapper(gym.Wrapper):
    def __init__(self, env, state_counter="none"):
        # intialize state counter
        self.state_counter = state_counter
        if self.state_counter != "none":
            self.state_count_dict = defaultdict(int)
        # this super() goes to the parent of the particular task, not to `object`
        super().__init__(env)

    def step(self, action):
        # add state counting to step function if desired
        step_return = self.env.step(action)
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
        obs = self.env.reset(wizkit_items=wizkit_items)
        if self.state_counter != "none":
            self.state_count_dict.clear()
            # current state counts as one visit
            obs.update(state_visits=np.array([1]))
        return obs
