import contextlib
import logging
import os
import random
import sqlite3
import time

DB = "ttyrecs.db"

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


@contextlib.contextmanager
def db(conn=None, filename=DB, new=False, rw=None, **kwargs):
    if conn:
        yield conn
        return
    try:
        conn = connect(filename, new, rw, **kwargs)
        yield conn
    finally:
        if conn:
            conn.close()


def connect(filename=DB, new=False, rw=None, **kwargs):
    assert new ^ os.path.exists(filename)
    if not new and not rw:
        filename += "?mode=ro"
    return sqlite3.connect("file:" + filename, uri=True, **kwargs)


def length(conn=None, count=False):
    with db(conn) as conn:
        if count:
            return conn.execute("SELECT COUNT(rowid) FROM ttyrecs").fetchone()[0]
        return conn.execute("SELECT MAX(rowid) FROM ttyrecs").fetchone()[0]


def ls(conn=None):
    with db(conn) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM meta")
        root, ctime, mtime = c.fetchone()
        logging.info(
            "root path: %s, ctime: %s, mtime: %s",
            root,
            time.ctime(ctime),
            time.ctime(mtime),
        )
        for row in c.execute("SELECT rowid, path FROM ttyrecs ORDER BY rowid"):
            yield row


def vacuum(conn=None):
    with db(conn=conn, rw=True) as conn:
        conn.execute("VACUUM")


def getrow(rowid="1", conn=None):
    with db(conn) as conn:
        result = conn.execute(
            "SELECT rowid, * FROM ttyrecs WHERE rowid = ?", (rowid,)
        ).fetchone()
        if result is None:
            raise ValueError("Row %s not found" % rowid)
        return result


def getmeta(conn=None):
    with db(conn) as conn:
        return conn.execute("SELECT * FROM meta").fetchone()


def getroot(dataset_name, conn=None):
    with db(conn) as conn:
        return conn.execute(
            "SELECT root FROM roots WHERE dataset_name=?", (dataset_name,)
        ).fetchone()


def setroot(dataset_name, root):
    root = os.path.abspath(root)
    with db(rw=True) as conn:
        conn.execute(
            "UPDATE roots SET root = ? WHERE dataset_name = ?", (root, dataset_name)
        )
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        conn.commit()


def getrandom(n=1, conn=None):
    with db(conn) as conn:
        last = length(conn)
        for _ in range(int(n)):
            rowid = random.randint(1, last)
            yield getrow(rowid, conn)


def getmostrecentgames(n=1, conn=None):
    with db(conn=conn) as conn:
        c = conn.execute("SELECT gameid FROM games ORDER BY gameid DESC LIMIT ?", (n,))
        return [g[0] for g in c.fetchall()]


def countgames(dataset_name, conn=None):
    with db(conn=conn) as conn:
        c = conn.execute(
            "SELECT COUNT(datasets.gameid) FROM datasets WHERE datasets.dataset_name=?",
            (dataset_name,),
        )
        return c.fetchone()[0]


def getgames(dataset_name, conn=None):
    with db(conn=conn) as conn:
        c = conn.execute(
            "SELECT games.name, games.gameid, games.starttime, games.endtime "
            "FROM games "
            "JOIN datasets ON games.gameid=datasets.gameid "
            "WHERE datasets.dataset_name=?",
            (dataset_name,),
        )
        for row in c:
            yield row


def addgames(dataset_name, *gameids, conn=None, commit=True):
    with db(conn, rw=True) as conn:
        conn.executemany(
            "INSERT INTO datasets VALUES (?, ?)",
            zip(gameids, [dataset_name] * len(gameids)),
        )
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def dropgames(dataset_name, *gameids, conn=None, commit=True):
    with db(conn, rw=True) as conn:
        conn.executemany(
            "DELETE FROM datasets WHERE gameid=? AND dataset_name=?",
            zip(gameids, [dataset_name] * len(gameids)),
        )
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def purgeemptygames(conn=None, commit=True):
    select = "SELECT DISTINCT(gameid) FROM ttyrecs"
    with db(conn, rw=True) as conn:
        conn.execute("DELETE FROM datasets WHERE NOT gameid IN (%s)" % select)
        conn.execute("DELETE FROM games WHERE NOT gameid IN (%s)" % select)
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def createdataset(dataset_name, root, conn=None, commit=True):
    with db(conn, rw=True) as conn:
        conn.execute("INSERT INTO roots VALUES (?, ?)", (dataset_name, root))
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def deletedataset(dataset_name, conn=None, commit=True):
    with db(conn, rw=True) as conn:
        conn.execute("DELETE datasets WHERE dataset_name=?", (dataset_name,))
        conn.execute("DELETE roots WHERE dataset_name=?", (dataset_name,))
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def create(filename=DB):
    ctime = time.time()

    with db(filename=filename, new=True) as conn:
        logging.info("Creating '%s' ...", DB)
        c = conn.cursor()

        c.execute(
            """CREATE TABLE meta
            (
                time REAL,
                mtime REAL
            )"""
        )
        c.execute("INSERT INTO meta VALUES (?, ?)", (ctime, ctime))

        c.execute(
            """CREATE TABLE ttyrecs
            (
                path        TEXT,
                part        INTEGER,
                size        INTEGER,
                mtime       REAL,
                gameid      INTEGER,
                PRIMARY KEY (gameid, part, path)
            )"""
        )

        c.execute(
            """CREATE TABLE games
            (
                gameid      INTEGER PRIMARY KEY,
                version     TEXT,
                points      INTEGER,
                deathdnum   INTEGER,
                deathlev    INTEGER,
                maxlvl      INTEGER,
                hp          INTEGER,
                maxhp       INTEGER,
                deaths      INTEGER,
                deathdate   INTEGER,
                birthdate   INTEGER,
                uid         INTEGER,
                role        TEXT,
                race        TEXT,
                gender      TEXT,
                align       TEXT,
                name        TEXT,
                death       TEXT,
                conduct     TEXT,
                turns       INTEGER,
                achieve     TEXT,
                realtime    INTEGER,
                starttime   INTEGER,
                endtime     INTEGER,
                gender0     TEXT,
                align0      TEXT,
                flags       TEXT
            )
            """
        )

        c.execute(
            """CREATE TABLE datasets
            (
                gameid        INTEGER,
                dataset_name  TEXT,
                PRIMARY KEY (gameid, dataset_name)
            )
            """
        )

        c.execute(
            """CREATE TABLE roots
            (
                dataset_name  TEXT PRIMARY KEY,
                root          TEXT
            )"""
        )

        conn.commit()
    logging.info(
        "Created Empty '%s'. Size: %.2f MB",
        DB,
        os.path.getsize(DB) / 1024**2,
    )
