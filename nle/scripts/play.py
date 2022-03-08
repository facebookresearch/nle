#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import ast
import contextlib
import os
import random
import termios
import time
import timeit
import tty

import gym

import nle  # noqa: F401
from nle import nethack

_ACTIONS = tuple(
    [nethack.MiscAction.MORE]
    + list(nethack.CompassDirection)
    + list(nethack.CompassDirectionLonger)
)


@contextlib.contextmanager
def dummy_context():
    yield None


@contextlib.contextmanager
def no_echo():
    tt = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)


def go_back(num_lines):
    print("\033[%dA" % num_lines)


def get_action(env, is_raw_env):
    if FLAGS.mode == "random":
        if not is_raw_env:
            action = env.action_space.sample()
        else:
            action = random.choice(_ACTIONS)
            print(action)
    elif FLAGS.mode == "human":
        while True:
            with no_echo():
                ch = ord(os.read(0, 1))
            if ch in [nethack.C("c")]:
                print("Received exit code {}. Aborting.".format(ch))
                return None
            try:
                if is_raw_env:
                    action = ch
                else:
                    action = env.actions.index(ch)
                break
            except ValueError:
                print(
                    ("Selected action '%s' is not in action list. Please try again.")
                    % chr(ch)
                )
                if not FLAGS.print_frames_separately:
                    print("\033[2A")  # Go up 2 lines.
                continue
    return action


def play():
    is_raw_env = FLAGS.env == "raw"

    if is_raw_env:
        if FLAGS.savedir is not None:
            os.makedirs(FLAGS.savedir, exist_ok=True)
            ttyrec = os.path.join(FLAGS.savedir, "nle.ttyrec.bz2")
        else:
            ttyrec = "/dev/null"
        env = nethack.Nethack(ttyrec=ttyrec, wizard=FLAGS.wizard)
    else:
        env = gym.make(
            FLAGS.env,
            save_ttyrec_every=2,
            savedir=FLAGS.savedir,
            max_episode_steps=FLAGS.max_steps,
            allow_all_yn_questions=True,
            allow_all_modes=True,
            wizard=FLAGS.wizard,
        )
        if FLAGS.seeds is not None:
            env.seed(FLAGS.seeds)

    obs = env.reset()

    steps = 0
    episodes = 0
    reward = 0.0
    action = None

    mean_sps = 0
    mean_reward = 0.0

    total_start_time = timeit.default_timer()
    start_time = total_start_time

    while True:
        if not FLAGS.no_render:
            if not is_raw_env:
                print("-" * 8 + " " * 71)
                print(f"Previous reward: {str(reward):64s}")
                act_str = repr(env.actions[action]) if action is not None else ""
                print(f"Previous action: {str(act_str):64s}")
                print("-" * 8)
                env.render(FLAGS.render_mode)
                print("-" * 8)
                print(obs["blstats"])
                if not FLAGS.print_frames_separately:
                    go_back(num_lines=33)
            else:
                print("Previous action:", action)
                obs = dict(zip(nle.nethack.OBSERVATION_DESC.keys(), obs))
                print(
                    nle.nethack.tty_render(
                        obs["tty_chars"], obs["tty_colors"], obs["tty_cursor"]
                    )
                )
                if not FLAGS.print_frames_separately:
                    go_back(num_lines=len(obs["tty_chars"]) + 3)

        action = get_action(env, is_raw_env)

        if action is None:
            break

        if is_raw_env:
            obs, done = env.step(action)
        else:
            obs, reward, done, info = env.step(action)
        steps += 1

        if is_raw_env:
            done = done or steps >= FLAGS.max_steps  # NLE does this by default.
        else:
            mean_reward += (reward - mean_reward) / steps

        if not done:
            continue

        time_delta = timeit.default_timer() - start_time

        if not is_raw_env:
            print("Final reward:", reward)
            print("End status:", info["end_status"].name)
            print("Mean reward:", mean_reward)

        sps = steps / time_delta
        print("Episode: %i. Steps: %i. SPS: %f" % (episodes, steps, sps))

        episodes += 1
        mean_sps += (sps - mean_sps) / episodes

        start_time = timeit.default_timer()

        steps = 0
        mean_reward = 0.0

        if episodes == FLAGS.ngames:
            break
        env.reset()
    env.close()
    print(
        "Finished after %i episodes and %f seconds. Mean sps: %f"
        % (episodes, timeit.default_timer() - total_start_time, mean_sps)
    )


def main():
    parser = argparse.ArgumentParser(description="NLE Play tool.")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enables debug mode, which will drop stack into "
        "an ipdb shell if an exception is raised.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="human",
        choices=["human", "random"],
        help="Control mode. Defaults to 'human'.",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="NetHackScore-v0",
        help="Gym environment spec. Defaults to 'NetHackStaircase-v0'.",
    )
    parser.add_argument(
        "-n",
        "--ngames",
        type=int,
        default=1,
        help="Number of games to be played before exiting. "
        "NetHack will auto-restart if > 1.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000_000,
        help="Number of maximum steps per episode.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Seeds to send to NetHack. Can be a dict or int. "
        "Defaults to None (no seeding).",
    )
    parser.add_argument(
        "--savedir",
        default="nle_data/play_data",
        help="Directory path where data will be saved. "
        "Defaults to 'nle_data/play_data'.",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disables env.render()."
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=["human", "full", "ansi"],
        help="Render mode. Defaults to 'human'.",
    )
    parser.add_argument(
        "--print-frames-separately",
        "-p",
        action="store_true",
        help="Don't overwrite frames, print them all.",
    )
    parser.add_argument(
        "--wizard",
        "-D",
        action="store_true",
        help="Use wizard mode.",
    )
    global FLAGS
    FLAGS = parser.parse_args()

    if FLAGS.debug:
        import ipdb

        cm = ipdb.launch_ipdb_on_exception
    else:
        cm = dummy_context

    with cm():
        if FLAGS.seeds is not None:
            # to handle both int and dicts
            FLAGS.seeds = ast.literal_eval(FLAGS.seeds)

        if FLAGS.savedir == "args":
            FLAGS.savedir = "{}_{}_{}.zip".format(
                time.strftime("%Y%m%d-%H%M%S"), FLAGS.mode, FLAGS.env
            )
        elif FLAGS.savedir == "None":
            FLAGS.savedir = None  # Not saving any ttyrecs.

        play()


if __name__ == "__main__":
    main()
