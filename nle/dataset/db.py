import contextlib
import glob
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


def vacuum():
    with db(rw=True) as conn:
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


def getroot(conn=None):
    root, _, _ = getmeta(conn)
    return root


def setroot(root):
    root = os.path.abspath(root)
    with db(rw=True) as conn:
        conn.execute("UPDATE meta SET root = ?, mtime = ?", (root, time.time()))
        conn.commit()


def getrandom(n=1, conn=None):
    with db(conn) as conn:
        last = length(conn)
        for _ in range(int(n)):
            rowid = random.randint(1, last)
            yield getrow(rowid, conn)


def markerror(rowid, rc, conn=None):
    with db(rw=True) as conn:
        conn.execute(
            "UPDATE ttyrecs SET is_error = ?, mtime = ? WHERE rowid = ?",
            (rc, time.time(), rowid),
        )
        conn.commit()


def update_batch(rowids, update_dicts, conn=None):
    assert len(rowids) == len(update_dicts)
    fields = update_dicts[0].keys()
    assert all(d.keys() == fields for d in update_dicts)

    field_string = ",".join([f"{field} = ?" for field in fields])
    sql_string = f"UPDATE ttyrecs SET {field_string}, mtime = ? WHERE rowid = ?"

    def gen():
        for rowid, update in zip(rowids, update_dicts):
            values = [update[f] for f in fields]
            yield tuple(values) + (time.time(), rowid)

    with db(conn, rw=True) as conn:
        conn.executemany(sql_string, gen())
        conn.commit()


def update(rowid, conn=None, **update_values):
    update_batch([rowid], [update_values], conn=conn)


def create(root):
    root = os.path.abspath(root)
    ctime = time.time()

    with db(new=True) as conn:
        logging.info("Creating '%s' ...", DB)
        c = conn.cursor()

        c.execute(
            """CREATE TABLE meta
            (root text, ctime real, mtime real)"""
        )
        c.execute("INSERT INTO meta VALUES (?,?,?)", (root, ctime, ctime))

        c.execute(
            """CREATE TABLE ttyrecs
            (
                path        TEXT,
                size        INTEGER,
                mtime       REAL,
                user        TEXT,
                frames      INTEGER DEFAULT -1,
                start_time  INTEGER,
                end_time    INTEGER,
                is_clean_bl INTEGER DEFAULT 0,
                is_error    INTEGER DEFAULT 0,
                gameid      INTEGER
            )"""
        )

        c.execute(
            """CREATE TABLE games
            (
                gameid      INTEGER PRIMARY KEY AUTOINCREMENT,
                user        TEXT,
                name        TEXT,
                points      INTEGER,
                race        TEXT,
                death       TEXT,
                turns       INTEGER,
                version     TEXT,
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
                gender      TEXT,
                align       TEXT,
                conduct     INTEGER,
                achieve     INTEGER,
                nconducts   INTEGER,
                nachieves   INTEGER,
                realtime    INTEGER,
                starttime   INTEGER,
                endtime     INTEGER,
                gamedelta   INTEGER,
                gender0     TEXT,
                align0      TEXT,
                flags       INTEGER
            )"""
        )

        def gen():
            for path in glob.iglob(root + "/*/*.ttyrec.bz2"):
                relpath = os.path.relpath(path, root)
                user = os.path.split(relpath)[0]
                yield (relpath, os.path.getsize(path), os.path.getmtime(path), user)

        c.executemany(
            "INSERT INTO ttyrecs (path, size, mtime, user) VALUES (?,?,?,?)", gen()
        )

        mtime = time.time()
        c.execute("UPDATE meta SET mtime = ?", (mtime,))

        conn.commit()

    logging.info(
        "Created '%s' in %.2f sec. Size: %.2f MB",
        DB,
        mtime - ctime,
        os.path.getsize(DB) / 1024**2,
    )


def sort():
    stime = time.time()
    with db(rw=True) as conn:
        c = conn.cursor()

        c.execute(
            """CREATE TABLE ordered_ttyrecs
            (
                path        TEXT,
                size        INTEGER,
                mtime       REAL,
                user        TEXT,
                frames      INTEGER DEFAULT -1,
                start_time  INTEGER,
                end_time    INTEGER,
                is_clean_bl INTEGER DEFAULT 0,
                is_error    INTEGER DEFAULT 0,
                gameid      INTEGER
            )
            """
        )
        c.execute(
            """INSERT INTO ordered_ttyrecs
            (
                path,
                size,
                mtime,
                user,
                frames,
                start_time,
                end_time,
                is_clean_bl,
                is_error,
                gameid
            )
            SELECT
                path,
                size,
                mtime,
                user,
                frames,
                start_time,
                end_time,
                is_clean_bl,
                is_error,
                gameid

            FROM ttyrecs ORDER BY path"""
        )
        c.execute("DROP TABLE ttyrecs")
        c.execute("ALTER TABLE ordered_ttyrecs RENAME TO ttyrecs")

        mtime = time.time()
        c.execute("UPDATE meta SET mtime = ?", (mtime,))

        conn.commit()

    vacuum()

    logging.info(
        "Sorted '%s' in %.2f sec. Size: %.2f MB",
        DB,
        time.time() - stime,
        os.path.getsize(DB) / 1024**2,
    )
