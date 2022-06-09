import contextlib
import logging
import os
import sqlite3
import time

DB = "ttyrecs.db"
logger = logging.getLogger("db")


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


def exists(filename=DB):
    return os.path.exists(filename)


def connect(filename=DB, new=False, rw=None, **kwargs):
    assert new ^ os.path.exists(filename)
    if not new and not rw:
        filename += "?mode=ro"
    return sqlite3.connect("file:" + filename, uri=True, **kwargs)


def ls(conn=None):
    with db(conn) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM meta")
        ctime, mtime = c.fetchone()
        logger.info(
            time.ctime(ctime),
            time.ctime(mtime),
        )
        for row in c.execute("SELECT rowid, path FROM ttyrecs ORDER BY rowid"):
            yield row


def vacuum(conn=None):
    with db(conn=conn, rw=True) as conn:
        conn.execute("VACUUM")


def get_row(rowid="1", conn=None):
    with db(conn) as conn:
        result = conn.execute(
            "SELECT rowid, * FROM ttyrecs WHERE rowid = ?", (rowid,)
        ).fetchone()
        if result is None:
            raise ValueError("Row %s not found" % rowid)
        return result


def get_meta(conn=None):
    with db(conn) as conn:
        return conn.execute("SELECT * FROM meta").fetchone()


def get_root(dataset_name, conn=None):
    with db(conn) as conn:
        return conn.execute(
            "SELECT root FROM roots WHERE dataset_name=?", (dataset_name,)
        ).fetchone()[0]


def set_root(dataset_name, root):
    root = os.path.abspath(root)
    with db(rw=True) as conn:
        conn.execute(
            "UPDATE roots SET root = ? WHERE dataset_name = ?", (root, dataset_name)
        )
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        conn.commit()


def get_ttyrec_version(dataset_name, conn=None):
    with db(conn) as conn:
        return conn.execute(
            "SELECT ttyrec_version FROM roots WHERE dataset_name=?", (dataset_name,)
        ).fetchone()[0]


def get_most_recent_games(n=1, conn=None):
    with db(conn=conn) as conn:
        c = conn.execute("SELECT gameid FROM games ORDER BY gameid DESC LIMIT ?", (n,))
        return [g[0] for g in c.fetchall()]


def count_games(dataset_name, conn=None):
    with db(conn=conn) as conn:
        c = conn.execute(
            "SELECT COUNT(datasets.gameid) FROM datasets WHERE datasets.dataset_name=?",
            (dataset_name,),
        )
        return c.fetchone()[0]


def get_games(dataset_name, conn=None):
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


def add_games(dataset_name, *gameids, conn=None, commit=True):
    with db(conn, rw=True) as conn:
        conn.executemany(
            "INSERT INTO datasets VALUES (?, ?)",
            zip(gameids, [dataset_name] * len(gameids)),
        )
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def drop_games(dataset_name, *gameids, conn=None, commit=True):
    with db(conn, rw=True) as conn:
        conn.executemany(
            "DELETE FROM datasets WHERE gameid=? AND dataset_name=?",
            zip(gameids, [dataset_name] * len(gameids)),
        )
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def delete_games_with_select(select, not_in=False, conn=None, commit=True):
    _not = "NOT" if not_in else ""
    with db(conn, rw=True) as conn:
        conn.execute("DELETE FROM ttyrecs WHERE %s gameid IN (%s)" % (_not, select))
        conn.execute("DELETE FROM datasets WHERE %s gameid IN (%s)" % (_not, select))
        conn.execute("DELETE FROM games WHERE %s gameid IN (%s)" % (_not, select))
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def create_dataset(dataset_name, root, ttyrec_version=0, conn=None, commit=True):
    with db(conn, rw=True) as conn:
        conn.execute(
            "INSERT INTO roots VALUES (?, ?, ?)", (dataset_name, root, ttyrec_version)
        )
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def delete_dataset(dataset_name, conn=None, commit=True):
    with db(conn, rw=True) as conn:
        conn.execute("DELETE datasets WHERE dataset_name=?", (dataset_name,))
        conn.execute("DELETE roots WHERE dataset_name=?", (dataset_name,))
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        if commit:
            conn.commit()


def create(filename=DB):
    ctime = time.time()

    with db(filename=filename, new=True) as conn:
        logger.info("Creating '%s' ...", DB)
        c = conn.cursor()

        c.execute(
            """CREATE TABLE meta
            (
                ctime REAL,
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
                PRIMARY KEY   (dataset_name, gameid)
            )
            """
        )

        c.execute(
            """CREATE TABLE roots
            (
                dataset_name   TEXT PRIMARY KEY,
                root           TEXT,
                ttyrec_version INTEGER
            )"""
        )

        conn.commit()
    logger.info(
        "Created Empty '%s'. Size: %.2f MB",
        filename,
        os.path.getsize(filename) / 1024**2,
    )
