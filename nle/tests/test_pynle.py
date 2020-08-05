import timeit
import random
import warnings

from nle import pynle


# MORE + compass directions + long compass directions.
ACTIONS = [
    13,
    107,
    108,
    106,
    104,
    117,
    110,
    98,
    121,
    75,
    76,
    74,
    72,
    85,
    78,
    66,
    89,
]


class TestPyNLE:
    def test_run_n_episodes(self, tmpdir, episodes=3):
        tmpdir.chdir()

        nle = pynle.NLE(observation_keys=("chars", "blstats"))
        chars, blstats = nle.reset()

        assert chars.shape == (21, 79)
        assert blstats.shape == (23,)

        nle.step(ord("y"))
        nle.step(ord("y"))
        nle.step(ord("\n"))

        steps = 0
        start_time = timeit.default_timer()
        start_steps = steps

        mean_sps = 0
        sps_n = 0

        for episode in range(episodes):
            while True:
                ch = random.choice(ACTIONS)
                _, done = nle.step(ch)
                if done:
                    break

                steps += 1

                if steps % 1000 == 0:
                    end_time = timeit.default_timer()
                    sps = (steps - start_steps) / (end_time - start_time)
                    sps_n += 1
                    mean_sps += (sps - mean_sps) / sps_n
                    print("%f SPS" % sps)
                    start_time = end_time
                    start_steps = steps
            print("Finished episode %i after %i steps." % (episode + 1, steps))
            nle.reset()

        print("Finished after %i steps. Mean sps: %f" % (steps, mean_sps))

        nethackdir = tmpdir.chdir()
        assert nethackdir.fnmatch("*nethackdir")
        assert tmpdir.ensure("nle.ttyrec")

        assert mean_sps > 10000

        if mean_sps < 15000:
            warnings.warn("Mean sps was only %f" % mean_sps)
