import json

import pytest  # NOQA: F401
from test_converter import getfilename
from test_db import conn  # NOQA: F401
from test_db import mockdata  # NOQA: F401

from nle import nethack

TTYRECS_TABLE_OFFSET = 0
GAMES_TABLE_OFFSET = 5
DATASETS_TABLE_OFFSET = 27 + GAMES_TABLE_OFFSET
ROOTS_TABLE_OFFSET = 2 + DATASETS_TABLE_OFFSET

TTYRECS_PATH_IDX = TTYRECS_TABLE_OFFSET + 0
TTYRECS_PART_IDX = TTYRECS_TABLE_OFFSET + 1
TTYRECS_MTIME_IDX = TTYRECS_TABLE_OFFSET + 3
TTYRECS_GAMEID_IDX = TTYRECS_TABLE_OFFSET + 4
GAMES_GAMEID_IDX = GAMES_TABLE_OFFSET + 0
GAMES_VERSION_IDX = GAMES_TABLE_OFFSET + 1
GAMES_ROLE_IDX = GAMES_TABLE_OFFSET + 12
GAMES_RACE_IDX = GAMES_TABLE_OFFSET + 13
GAMES_GEN_IDX = GAMES_TABLE_OFFSET + 14
GAMES_ALIGN_IDX = GAMES_TABLE_OFFSET + 15
GAMES_NAME_IDX = GAMES_TABLE_OFFSET + 16
GAMES_DEATH_IDX = GAMES_TABLE_OFFSET + 17
TTYREC_VERSION_IDX = ROOTS_TABLE_OFFSET + 2

DATASETS_GAMEID_IDX = DATASETS_TABLE_OFFSET + 0


class TestPopulateDB:
    def test_dump_altorg_db(self, conn):  # NOQA: F811
        # Test that populating with a mock altorg directory generates
        # the right database (effectively an integration test)
        with open(getfilename("altorg/db.json"), "r") as f:
            dump = json.load(f)

        cmd = """
        SELECT * FROM ttyrecs
        INNER JOIN games ON games.gameid=ttyrecs.gameid
        INNER JOIN datasets ON games.gameid=datasets.gameid
        INNER JOIN roots ON roots.dataset_name=datasets.dataset_name
        WHERE datasets.dataset_name='altorgtest'
        ORDER BY gameid
        """
        result = conn.execute(cmd).fetchall()
        assert len(result) == len(dump)
        # We expect repeats of the dataset generation to insert
        # games and ttyrecs in the right order into the database.
        # Since the database may already have a different dataset,
        # the gameids may be offset, so we check the relative order:
        # EG: if expected (3, 3, 4, 5, 5) actual can be (11, 11, 12, 13, 13)
        actual_gameid_offset = result[0][TTYRECS_GAMEID_IDX] - 1
        expected_gameid_offset = dump[0][TTYRECS_GAMEID_IDX] - 1
        offset = actual_gameid_offset - expected_gameid_offset
        for actual, expected in zip(result, dump):
            expected[TTYRECS_MTIME_IDX] = actual[TTYRECS_MTIME_IDX]
            assert type(actual[TTYRECS_MTIME_IDX]) == float
            expected[TTYRECS_GAMEID_IDX] += offset
            expected[GAMES_GAMEID_IDX] += offset
            expected[DATASETS_GAMEID_IDX] += offset

            assert actual == tuple(expected)

    def test_dump_nle_db(self, conn):  # NOQA: F811
        # Test that populating with an nle_data directory generates
        # the right database (effectively an integration test)

        cmd = """
        SELECT * FROM ttyrecs
        INNER JOIN games ON games.gameid=ttyrecs.gameid
        INNER JOIN datasets ON games.gameid=datasets.gameid
        INNER JOIN roots ON roots.dataset_name=datasets.dataset_name
        WHERE datasets.dataset_name='nletest'
        ORDER BY gameid
        """
        result = conn.execute(cmd).fetchall()
        endings = [
            ".0.ttyrec%i.bz2",
            ".2.ttyrec%i.bz2",
            ".4.ttyrec%i.bz2",
            ".0.ttyrec%i.bz2",
            ".2.ttyrec%i.bz2",
            ".4.ttyrec%i.bz2",
        ]
        endings = [e % nethack.TTYREC_VERSION for e in endings]
        assert len(result) == 6

        paths = []
        for i, actual in enumerate(result):
            paths.append(actual[TTYRECS_PATH_IDX])

            assert actual[TTYRECS_PATH_IDX].endswith(endings[i])
            assert actual[TTYRECS_PART_IDX] == 0
            assert actual[TTYRECS_GAMEID_IDX] == result[0][TTYRECS_GAMEID_IDX] + i
            assert actual[GAMES_VERSION_IDX] == "3.6.6"
            assert actual[GAMES_ROLE_IDX] == "Mon"
            assert actual[GAMES_RACE_IDX] == "Hum"
            assert actual[GAMES_GEN_IDX] == "Mal"
            assert actual[GAMES_ALIGN_IDX] == "Neu"
            assert actual[GAMES_NAME_IDX] == "Agent"
            assert actual[GAMES_DEATH_IDX] == "escaped"
            assert actual[TTYREC_VERSION_IDX] == nethack.TTYREC_VERSION

        assert paths == sorted(paths)
