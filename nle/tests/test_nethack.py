# Copyright (c) Facebook, Inc. and its affiliates.
import timeit
import random
import warnings

import numpy as np

import pytest

from nle import nethack, _pynethack


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


class TestNetHack:
    @pytest.fixture
    def game(self):  # Make sure we close even on test failure.
        g = nethack.Nethack(observation_keys=("chars", "blstats"))
        try:
            yield g
        finally:
            g.close()

    def test_close_and_restart(self):
        game = nethack.Nethack()
        game.reset()
        game.close()

        game = nethack.Nethack()
        game.reset()
        game.close()

    def test_run_n_episodes(self, tmpdir, game, episodes=3):
        olddir = tmpdir.chdir()

        chars, blstats = game.reset()

        assert chars.shape == (21, 79)
        assert blstats.shape == (25,)

        game.step(ord("y"))
        game.step(ord("y"))
        game.step(ord("\n"))

        steps = 0
        start_time = timeit.default_timer()
        start_steps = steps

        mean_sps = 0
        sps_n = 0

        for episode in range(episodes):
            while True:
                ch = random.choice(ACTIONS)
                _, done = game.step(ch)
                if done:
                    # This will typically be DIED, but could be POISONED, etc.
                    assert int(game.how_done()) < int(nethack.GENOCIDED)
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
            game.reset()

        print("Finished after %i steps. Mean sps: %f" % (steps, mean_sps))

        nethackdir = tmpdir.chdir()

        assert nethackdir.fnmatch("nle*")
        assert tmpdir.ensure("nle.ttyrec")

        if mean_sps < 15000:
            warnings.warn("Mean sps was only %f" % mean_sps)
        olddir.chdir()
        # No call to game.close() as fixture will do that for us.

    def test_several_nethacks(self, game):
        game.reset()
        game1 = nethack.Nethack()
        game1.reset()

        try:
            for _ in range(300):
                ch = random.choice(ACTIONS)
                _, done = game.step(ch)
                if done:
                    game.reset()
                _, done = game1.step(ch)
                if done:
                    game1.reset()
        finally:
            game1.close()

    def test_set_initial_seeds(self):
        if not nethack.NLE_ALLOW_SEEDING:
            return  # Nothing to test.
        game = nethack.Nethack(copy=True)
        game.set_initial_seeds(core=42, disp=666)
        obs0 = game.reset()
        try:
            seeds0 = game.get_current_seeds()
            game.set_initial_seeds(core=42, disp=666)
            obs1 = game.reset()
            seeds1 = game.get_current_seeds()
            np.testing.assert_equal(obs0, obs1)
            assert seeds0 == seeds1
        finally:
            game.close()

    def test_set_seed_after_reset(self, game):
        if not nethack.NLE_ALLOW_SEEDING:
            return  # Nothing to test.

        game.reset()
        # Could fail on a system without a good source of randomness:
        assert game.get_current_seeds()[2] is True
        game.set_current_seeds(core=42, disp=666)
        assert game.get_current_seeds() == (42, 666, False)


class TestNetHackFurther:
    def test_run(self):
        # TODO: Implement ttyrecording filename in libnethack wrapper.
        # archivefile = tempfile.mktemp(suffix="nethack_test", prefix=".zip")

        game = nethack.Nethack(
            observation_keys=("glyphs", "chars", "colors", "blstats", "program_state")
        )
        _, _, _, _, program_state = game.reset()
        actions = [
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
            nethack.MiscAction.MORE,
        ]

        for action in actions:
            while not program_state[3]:  # in_moveloop.
                obs, done = game.step(nethack.MiscAction.MORE)
                _, _, _, _, program_state = obs

            obs, done = game.step(action)
            if done:
                # Only the good die young.
                obs = game.reset()

            glyphs, chars, colors, blstats, _ = obs

            x, y = blstats[:2]

            assert np.count_nonzero(chars == ord("@")) == 1

            # That's where you're @.
            assert chars[y, x] == ord("@")

            # You're bright (4th bit, 8) white (7), too.
            assert colors[y, x] == 8 ^ 7

            mon = nethack.permonst(nethack.glyph_to_mon(glyphs[y][x]))
            assert mon.mname == "monk"
            assert mon.mlevel == 10

            class_sym = nethack.class_sym.from_mlet(mon.mlet)
            assert class_sym.sym == "@"
            assert class_sym.explain == "human or elf"

        game.close()

    def test_illegal_filename(self):
        with pytest.raises(IOError):
            nethack.Nethack(ttyrec="")
        game = nethack.Nethack()
        with pytest.raises(IOError):
            game.reset("")


class TestNethackSomeObs:
    @pytest.fixture
    def game(self):  # Make sure we close even on test failure.
        g = nethack.Nethack(observation_keys=("program_state", "message", "internal"))
        try:
            yield g
        finally:
            g.close()

    def test_message(self, game):
        messages = []

        program_state, message, _ = game.reset()
        messages.append(message)
        while not program_state[3]:  # in_moveloop.
            (program_state, message, _), done = game.step(nethack.MiscAction.MORE)
            messages.append(message)

        greeting = (
            b"Hello Agent, welcome to NetHack!  You are a neutral male human Monk."
        )
        saw_greeting = True
        for message in messages:
            # `greeting` is often the last message, but not always -- e.g.,
            # it could also be "Be careful!  New moon tonight.".
            assert len(message) == 256
            if (
                memoryview(message)[: len(greeting)] == greeting
                and memoryview(message)[len(greeting)] == 0
            ):
                saw_greeting = True
        assert saw_greeting

    def test_internal(self, game):
        program_state, _, internal = game.reset()
        while not program_state[3]:  # in_moveloop.
            # We are not in moveloop. We might still be in "normal game"
            # if something_worth_saving is true. Example: Startup with
            # "--More--", "Be careful!  New moon tonight."
            assert game.in_normal_game() == program_state[5]  # something_worth_saving.
            (program_state, _, internal), done = game.step(nethack.MiscAction.MORE)

        assert game.in_normal_game()
        assert internal[0] == 1  # deepest_lev_reached.

        (_, _, internal), done = game.step(nethack.Command.INVENTORY)
        assert internal[3] == 1  # xwaitforspace


def get_object(name):
    for index in range(nethack.NUM_OBJECTS):
        obj = nethack.objclass(index)
        if nethack.OBJ_NAME(obj) == name:
            return obj
    else:
        raise ValueError("'%s' not found!" % name)


class TestNethackFunctionsAndConstants:
    def test_permonst_and_class_sym(self):
        glyph = 155  # Lichen.

        mon = nethack.permonst(nethack.glyph_to_mon(glyph))

        assert mon.mname == "lichen"

        cs = nethack.class_sym.from_mlet(mon.mlet)

        assert cs.sym == "F"
        assert cs.explain == "fungus or mold"

        assert nethack.NHW_MESSAGE == 1
        assert hasattr(nethack, "MAXWIN")

    def test_permonst(self):
        mon = nethack.permonst(0)
        assert mon.mname == "giant ant"
        del mon

        mon = nethack.permonst(1)
        assert mon.mname == "killer bee"

    def test_some_constants(self):
        assert nethack.GLYPH_MON_OFF == 0
        assert nethack.NUMMONS > 300

    def test_illegal_numbers(self):
        with pytest.raises(
            IndexError,
            match=r"should be between 0 and NUMMONS \(%i\) but got %i"
            % (nethack.NUMMONS, nethack.NUMMONS),
        ):
            nethack.permonst(nethack.NUMMONS)

        with pytest.raises(
            IndexError,
            match=r"should be between 0 and NUMMONS \(%i\) but got %i"
            % (nethack.NUMMONS, -1),
        ):
            nethack.permonst(-1)

        with pytest.raises(
            IndexError,
            match=r"should be between 0 and MAXMCLASSES \(%i\) but got 127"
            % nethack.MAXMCLASSES,
        ):
            nethack.class_sym.from_mlet("\x7F")

    def test_objclass(self):
        obj = nethack.objclass(0)
        assert nethack.OBJ_NAME(obj) == "strange object"

        food_ration = get_object("food ration")
        assert food_ration.oc_weight == 20

        elven_dagger = get_object("elven dagger")
        assert nethack.OBJ_DESCR(elven_dagger) == "runed dagger"

    def test_objdescr(self):
        od = nethack.objdescr.from_idx(0)
        assert od.oc_name == "strange object"
        assert od.oc_descr is None

        elven_dagger = get_object("elven dagger")
        od = nethack.objdescr.from_idx(elven_dagger.oc_name_idx)
        assert od.oc_name == "elven dagger"
        assert od.oc_descr == "runed dagger"

        # Example of how to do this with glyphs.
        glyph = nethack.GLYPH_OBJ_OFF + elven_dagger.oc_name_idx
        idx = nethack.glyph_to_obj(glyph)
        assert idx == elven_dagger.oc_name_idx
        assert nethack.objdescr.from_idx(idx) is od


class TestNethackGlanceObservation:
    @pytest.fixture
    def game(self):  # Make sure we close even on test failure.
        g = nethack.Nethack(playername="MonkBot-mon-hum-neu-mal")
        try:
            yield g
        finally:
            g.close()

    def test_new_observation_shapes(self, game):
        game = nethack.Nethack()
        game.reset()

        screen_description_buff = game._obs_buffers["screen_descriptions"]
        glyph_buff = game._obs_buffers["glyphs"]
        glance_shape = screen_description_buff.shape
        assert glyph_buff.shape == glance_shape[:2]
        assert _pynethack.nethack.NLE_SCREEN_DESCRIPTION_LENGTH == glance_shape[-1]
        assert len(glance_shape) == 3

    def test_glance_descriptions(self, game):
        game.reset()

        # rather naughty - testing against private impl
        glyph_buff = game._obs_buffers["glyphs"]
        char_buff = game._obs_buffers["chars"]
        desc_buff = game._obs_buffers["screen_descriptions"]

        row, col = glyph_buff.shape
        episodes = 6
        for _ in range(episodes):
            game.reset()
            for i in range(row):
                for j in range(col):
                    glyph = glyph_buff[i][j]
                    char = char_buff[i][j]
                    letter = chr(char)
                    glance = "".join(chr(c) for c in desc_buff[i][j] if c != 0)
                    if char == 32:  # no text
                        assert glance == ""
                        assert (desc_buff[i][j] == 0).all()
                    elif glyph == 2378 and letter == ".":
                        assert glance == "floor of a room"
                    elif glyph == 333 and letter == "@":  # us!
                        assert glance == "human monk called MonkBot"
                    elif glyph == 413:  # pet cat
                        assert glance == "tame kitten"
                    elif glyph == 397:  # pet dog
                        assert glance == "tame little dog"
                    elif letter in "-":  # illustrate same char, diff descrip
                        if glyph == 2378:
                            assert glance == "grave"
                        elif glyph == 2363:
                            assert glance == "wall"
                        elif glyph == 2372:
                            assert glance == "open door"


class TestNethackTerminalObservation:
    @pytest.fixture
    def game(self):  # Make sure we close even on test failure.
        g = nethack.Nethack(playername="MonkBot-mon-hum-neu-mal")
        try:
            yield g
        finally:
            g.close()

    def test_new_observation_shapes(self, game):
        game.reset()

        tty_chars = game._obs_buffers["tty_chars"]
        tty_colors = game._obs_buffers["tty_colors"]
        tty_cursor = game._obs_buffers["tty_cursor"]
        assert tty_colors.shape == tty_chars.shape
        assert tty_cursor.shape == (2,)
        terminal_shape = tty_chars.shape
        assert _pynethack.nethack.NLE_TERM_LI == terminal_shape[0]
        assert _pynethack.nethack.NLE_TERM_CO == terminal_shape[1]

    def test_observations(self, game):
        game.reset()

        tty_chars = game._obs_buffers["tty_chars"]
        tty_colors = game._obs_buffers["tty_colors"]

        top_line = "".join(chr(c) for c in tty_chars[0])
        bottom_sub1_line = "".join(chr(c) for c in tty_chars[-2])
        bottom_line = "".join(chr(c) for c in tty_chars[-1])
        assert top_line.startswith(
            "Hello MonkBot, welcome to NetHack!  You are a neutral male human Monk."
        )
        assert bottom_sub1_line.startswith("MonkBot the Candidate")
        assert bottom_line.startswith("Dlvl:1")

        for c, font in zip(tty_chars.reshape(-1), tty_colors.reshape(-1)):
            if chr(c) == "@":
                assert font == 15  # BRIGHT_WHITE
            if chr(c) == " ":
                assert font == 0  # NO_COLOR

    def test_crop(self, game):
        game.reset()

        g_chars = game._obs_buffers["chars"].reshape(-1)
        g_cols = game._obs_buffers["colors"].reshape(-1)

        # DUNGEON is [21, 79], TTY is [24, 80]. Crop as follows  to get alignment.
        t_chars = game._obs_buffers["tty_chars"][1:-2, :-1].reshape(-1)
        t_cols = game._obs_buffers["tty_colors"][1:-2, :-1].reshape(-1)

        for g_char, g_col, t_char, t_col in zip(g_chars, g_cols, t_chars, t_cols):
            assert g_char == t_char
            assert g_col == t_col


class TestNethackMiscObservation:
    @pytest.fixture
    def game(self):  # Make sure we close even on test failure.
        g = nethack.Nethack(playername="MonkBot-mon-hum-neu-mal")
        try:
            yield g
        finally:
            g.close()

    def test_misc_yn_question(self, game):
        game.reset()
        misc = game._obs_buffers["misc"]
        internal = game._obs_buffers["internal"]
        if misc[2]:
            game.step(ord(" "))

        assert np.all(misc == 0)
        np.testing.assert_array_equal(misc, internal[1:4])

        game.step(nethack.M("p"))  # pray
        np.testing.assert_array_equal(misc, np.array([1, 0, 0]))
        np.testing.assert_array_equal(misc, internal[1:4])

        game.step(ord("n"))
        assert np.all(misc == 0)
        np.testing.assert_array_equal(misc, internal[1:4])

    def test_misc_getline(self, game):
        game.reset()
        misc = game._obs_buffers["misc"]
        internal = game._obs_buffers["internal"]
        if misc[2]:
            game.step(ord(" "))

        assert np.all(misc == 0)
        np.testing.assert_array_equal(misc, internal[1:4])

        game.step(nethack.M("n"))  # name ..
        game.step(ord("a"))  # ... the current level
        np.testing.assert_array_equal(misc, np.array([0, 1, 0]))
        np.testing.assert_array_equal(misc, internal[1:4])

        for let in "Gehennom":
            game.step(ord(let))
            np.testing.assert_array_equal(misc, np.array([0, 1, 0]))
            np.testing.assert_array_equal(misc, internal[1:4])

        game.step(ord("\r"))
        assert np.all(misc == 0)
        np.testing.assert_array_equal(misc, internal[1:4])

    def test_misc_wait_for_space(self, game):
        game.reset()
        misc = game._obs_buffers["misc"]
        internal = game._obs_buffers["internal"]
        if misc[2]:
            game.step(ord(" "))

        assert np.all(misc == 0)
        np.testing.assert_array_equal(misc, internal[1:4])

        game.step(ord("i"))
        np.testing.assert_array_equal(misc, np.array([0, 0, 1]))
        np.testing.assert_array_equal(misc, internal[1:4])

        game.step(ord(" "))
        assert np.all(misc == 0)
        np.testing.assert_array_equal(misc, internal[1:4])
