import contextlib
import glob
import logging
import os
import random
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from datetime import timezone

DB = "ttyrecs.db"

XLOGFILE_COLUMNS = [
    ("version", str),
    ("points", int),
    ("deathdnum", int),
    ("deathlev", int),
    ("maxlvl", int),
    ("hp", int),
    ("maxhp", int),
    ("deaths", int),
    ("deathdate", int),
    ("birthdate", int),
    ("uid", int),
    ("role", str),
    ("race", str),
    ("gender", str),
    ("align", str),
    ("name", str),
    ("death", str),
    ("conduct", str),
    ("turns", int),
    ("achieve", str),
    ("realtime", int),
    ("starttime", int),
    ("endtime", int),
    ("gender0", str),
    ("align0", str),
    ("flags", str),
]

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


def assign_ttyrecs_to_games(ttyrecs, games):
    assigned = []  # (path, file_starttime, gameid, game_starttime, game_endtime)
    for t in ttyrecs:
        s_time = t.split("/")[-1][:-11]
        try:
            s_time = (
                datetime.fromisoformat(s_time).replace(tzinfo=timezone.utc).timestamp()
            )
        except ValueError:
            logging.info("Failed to process: '%s'" % t)
            continue
        assigned.append([t, s_time, -1, -1, -1])
    assigned.sort(key=lambda x: x[1])
    games.sort(key=lambda x: x[1])

    gg, tt = 0, 0
    while gg < len(games) and tt < len(assigned):
        if assigned[tt][1] > games[gg][2]:
            gg += 1
        else:
            assigned[tt][2] = games[gg][0]
            assigned[tt][3] = games[gg][1]
            assigned[tt][4] = games[gg][2]
            tt += 1

    return [(ttyrec, gameid) for ttyrec, _, gameid, _, _ in assigned if gameid > 0]


def add_altorg_directory(path, name, filename=DB):
    with db(filename=filename, rw=True) as conn:

        logging.info("Adding dataset '%s' ('%s') to '%s' " % (name, path, filename))
        root = os.path.abspath(path)
        stime = time.time()

        c = conn.cursor()

        createdataset(name, root, conn=c, commit=False)

        for xlogfile in glob.iglob(path + "/xlogfile.*"):
            sep = ":" if xlogfile.endswith(".txt") else "\t"
            game_gen = game_data_generator(xlogfile, separator=sep)
            insert_sql = f"""
                INSERT INTO games
                VALUES (NULL, {','.join('?' for _ in XLOGFILE_COLUMNS)} )
            """
            c.executemany(insert_sql, game_gen)

            gameids = getmostrecentgames(c.rowcount, conn=c)
            addgames(name, *gameids, conn=c, commit=False)
            logging.info("Found %i games in '%s'" % (len(gameids), xlogfile))

        ttyrecs_dict = defaultdict(list)
        for ttyrec in glob.iglob(path + "/*/*.ttyrec.bz2"):
            ttyrecs_dict[ttyrec.split("/")[-2].lower()].append(ttyrec)

        games_dict = defaultdict(list)
        for pname, gameid, start, end in getgames(name, conn=c):
            games_dict[pname.lower()].append((gameid, start, end))

        logging.info("Matching up ttyrecs to games...")

        empty_games = []
        for pname in ttyrecs_dict.keys():
            assigned = assign_ttyrecs_to_games(ttyrecs_dict[pname], games_dict[pname])
            if assigned:
                ttyrecs, gameids = zip(*assigned)
                ttyrec_gen = ttyrec_data_generator(ttyrecs, gameids, root)
                c.executemany("INSERT INTO ttyrecs VALUES (?,?,?,?,?)", ttyrec_gen)
            elif games_dict[pname]:
                empty_games.extend(gid for gid, _, _ in games_dict[pname])
        for pname in games_dict:
            if pname not in ttyrecs_dict:
                empty_games.extend(gid for gid, _, _ in games_dict[pname])

        purgeemptygames(conn=c, commit=False)

        mtime = time.time()
        c.execute("UPDATE meta SET mtime = ?", (mtime,))

        conn.commit()

        logging.info("Optimizing DB...")

        vacuum(conn=conn)
        games_added = countgames(name, conn=conn)

    logging.info(
        "Updated '%s' in %.2f sec. Size: %.2f MB, Games: %i",
        DB,
        mtime - stime,
        os.path.getsize(DB) / 1024**2,
        games_added,
    )


def add_nledata_directory(path, name, filename=DB):
    with db(filename=filename, rw=True) as conn:

        logging.info("Adding dataset '%s' ('%s') to '%s' " % (name, path, filename))
        root = os.path.abspath(path)
        stime = time.time()

        c = conn.cursor()

        createdataset(name, root, conn=c, commit=False)

        for xlogfile in glob.iglob(path + "/*/*.xlogfile"):

            stem = xlogfile.replace(".xlogfile", ".*.ttyrec.bz2")
            resets = set(int(i.split(".")[-3]) for i in glob.iglob(stem))

            def filter(gen):
                # NB: xlogfile may have more rows than in directory
                #     due to 'save_ttyrec_every' option in env.py
                for line_no, line in enumerate(gen):
                    if line_no in resets:
                        yield line

            game_gen = game_data_generator(xlogfile, filter=filter)
            insert_sql = f"""
                INSERT INTO games
                VALUES (NULL, {','.join('?' for _ in XLOGFILE_COLUMNS)} )
            """
            c.executemany(insert_sql, game_gen)

            gameids = getmostrecentgames(c.rowcount, conn=c)
            addgames(name, *gameids, conn=conn, commit=False)

            valid_resets = list(resets)[: len(gameids)]
            ttyrecs = [
                stem.replace("*", str(r)) for r in sorted(valid_resets, reverse=True)
            ]
            ttyrec_gen = ttyrec_data_generator(ttyrecs, gameids, root)
            c.executemany("INSERT INTO ttyrecs VALUES (?,?,?,?,?)", ttyrec_gen)

        mtime = time.time()
        c.execute("UPDATE meta SET mtime = ?", (mtime,))

        conn.commit()
        games_added = countgames(name, conn=conn)

    logging.info(
        "Updated '%s' in %.2f sec. Size: %.2f MB, Games: %i",
        filename,
        mtime - stime,
        os.path.getsize(filename) / 1024**2,
        games_added,
    )


def ttyrec_data_generator(ttyrecs, gameids, root):
    last_gameid = None
    for path, gameid in zip(ttyrecs, gameids):
        if gameid != last_gameid:
            part = 0
        relpath = os.path.relpath(path, root)
        yield (
            relpath,
            part,
            os.path.getsize(path),
            os.path.getmtime(path),
            gameid,
        )
        part += 1
        last_gameid = gameid


def game_data_generator(xlogfile, filter=lambda x: x, separator="\t"):
    with open(xlogfile, "rb") as f:
        for line in filter(f.readlines()):
            game_data = defaultdict(lambda: -1)
            for words in line.decode("latin-1").split(separator):
                key, *var = words.split("=")
                game_data[key] = "=".join(var)

            if "while" in game_data:
                game_data["death"] += " while " + game_data["while"]

            yield tuple(ctype(game_data[key]) for key, ctype in XLOGFILE_COLUMNS)
