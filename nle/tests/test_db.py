import os
import time

import pytest
import test_converter

from nle.dataset import db

TTYRECS = [
    ("aaa", test_converter.TTYREC_2020),
    ("bbb", test_converter.TTYREC_IBMGRAPHICS),
    ("ccc", test_converter.TTYREC_2020),
]


@pytest.fixture(scope="session")
def mockdata(tmpdir_factory):  # Create mock data.
    """This fixture needs to be imported to generate a fake db.

    TEST DB structure: (9 entries)
    * 3 users  *"aaa", "bbb", "ccc",
    * each user with 3 copies of the same ttyrec ("[aaa|bbb|ccc].ttyrec.bz2")
    * user "aaa" and "ccc" have the same ttyrecs, "bbb"'s is different
    * user "ccc" has all of their ttyrecs registered as just one "game"/episode
    """
    basetemp = tmpdir_factory.getbasetemp()

    for user, ttyrec in TTYRECS:
        if basetemp.join(user).exists():
            break
        with open(test_converter.getfilename(ttyrec), "rb") as f:
            rec = f.read()
        d = tmpdir_factory.mktemp(user, numbered=False)
        for fn, _ in TTYRECS:
            fn = d.join(fn + ".ttyrec.bz2")
            fn.ensure()
            fn.write(rec, "wb")

    oldcwd = os.getcwd()
    newcwd = tmpdir_factory.getbasetemp()
    try:
        os.chdir(newcwd)
        if not newcwd.join("ttyrecs.db").exists():
            db.create(".")
            db.sort()
            with db.db(rw=True) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO games (user, name, death, points) VALUES (?,?,?,?)",
                    ("ccc", "cc the winner", "ascended", 999),
                )
                c.execute(
                    "UPDATE ttyrecs SET gameid =? WHERE user = ?", (c.lastrowid, "ccc")
                )
                conn.commit()
        yield tmpdir_factory.getbasetemp()
    finally:
        os.chdir(oldcwd)


@pytest.fixture(scope="session")
def conn(mockdata):
    """This fixture needs to be imported to receive a connection to working db."""
    with db.db() as conn:
        yield conn


class TestDB:
    def test_conn(self, conn):
        assert conn

    def test_length(self):
        assert db.length() == 9

    def test_ls(self):
        for i, row in enumerate(db.ls()):
            rowid, path = row
            assert int(rowid) == i + 1
        assert i + 1 == 9

    def test_vacuum(self):
        db.vacuum()

    def test_getrow(self):
        row1 = db.getrow("1")
        row5 = db.getrow("5")
        row8 = db.getrow("8")

        assert len(row8) == 11
        rowids, paths, sizes, ctimes, *_ = zip(row1, row5, row8)
        assert rowids == (1, 5, 8)
        # Order of glob (via os.listdir) OS dependent.
        names, ttyrecs = list(zip(*TTYRECS))
        for p in paths:
            assert os.path.dirname(p) in names
            assert os.path.basename(p) in [n + ".ttyrec.bz2" for n in names]
        assert sizes == (58697, 1779, 58697)

    def test_getmeta(self, conn):
        with conn:
            meta = db.getmeta(conn)
        root, _, mtime = meta

        assert root == os.getcwd()
        assert root == db.getroot()
        assert 0 < time.time() - mtime < 30

    def test_getrandom(self):
        for _i, row in enumerate(db.getrandom(1)):
            assert len(row) == 11
        assert _i == 0

        for _i, row in enumerate(db.getrandom(5)):
            assert len(row) == 11
        assert _i == 4

    def test_setroot(self):
        oldroot = db.getroot()
        newroot = "/my/new/root"
        db.setroot(newroot)
        assert db.getroot() == newroot
        db.setroot(oldroot)
        assert db.getroot() == oldroot
