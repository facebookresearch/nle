import collections
import datetime
import glob
import logging
import os
import time

from nle.dataset import db

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

FIVE_MINS = 5 * 60


def altorg_filename_to_timestamp(filename):
    ts = filename.split("/")[-1][:-11]
    try:
        ts = datetime.datetime.fromisoformat(ts)
    except ValueError:
        logging.info("Skipping: '%s'" % filename)
        return -1
    return ts.replace(tzinfo=datetime.timezone.utc).timestamp()


def assign_ttyrecs_to_games(ttyrecs, games):
    assigned = []  # (path, file_creationtime, gameid, game_starttime, game_endtime)
    for t in ttyrecs:
        s_time = altorg_filename_to_timestamp(t)
        assigned.append([t, s_time, -1, -1, -1])

    # We sort games and ttyrecs by start/creationtime and will pair them in one pass.
    # NB: We only want ttyrecs that can be assigned to a game.
    assigned.sort(key=lambda x: x[1])
    games.sort(key=lambda x: x[1])

    gg, tt = 0, 0
    while gg < len(games) and tt < len(assigned):
        if assigned[tt][1] > games[gg][2]:
            # This ttyrec was created after this game ends, so check next game.
            gg += 1
        elif assigned[tt][1] < games[gg][1] - FIVE_MINS:
            # We allow a 5min window, since the ttyrec creation time and xlogfile
            # starttime can differ slightly. However ttyrec starts well before
            # this game start, so check next ttyrec.
            assigned[tt][2] = 0
            assigned[tt][3] = games[gg][1]
            assigned[tt][4] = games[gg][2]
            tt += 1
        else:
            # This ttyrec starts after the game starts ( - 5 mins buffer)
            # and ends before this game ends, so we can assign
            assigned[tt][2] = games[gg][0]
            assigned[tt][3] = games[gg][1]
            assigned[tt][4] = games[gg][2]
            tt += 1
    return [(ttyrec, gameid) for ttyrec, _, gameid, _, _ in assigned if gameid != -1]


def add_altorg_directory(path, name, filename=db.DB):
    with db.db(filename=filename, rw=True) as conn:

        logging.info("Adding dataset '%s' ('%s') to '%s' " % (name, path, filename))
        root = os.path.abspath(path)
        stime = time.time()

        c = conn.cursor()

        db.create_dataset(name, root, conn=c, commit=False)

        for xlogfile in reversed(sorted(glob.iglob(path + "/xlogfile.*"))):
            sep = ":" if xlogfile.endswith(".txt") else "\t"
            game_gen = game_data_generator(xlogfile, separator=sep)
            insert_sql = f"""
                INSERT INTO games
                VALUES (NULL, {','.join('?' for _ in XLOGFILE_COLUMNS)} )
            """
            c.executemany(insert_sql, game_gen)

            gameids = db.get_most_recent_games(c.rowcount, conn=c)
            db.add_games(name, *gameids, conn=c, commit=False)
            logging.info("Found %i games in '%s'" % (len(gameids), xlogfile))

        ttyrecs_dict = collections.defaultdict(list)
        for ttyrec in glob.iglob(path + "/*/*.ttyrec.bz2"):
            ttyrecs_dict[ttyrec.split("/")[-2].lower()].append(ttyrec)

        games_dict = collections.defaultdict(list)
        for pname, gameid, start, end in db.get_games(name, conn=c):
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

        db.purge_empty_games(conn=c, commit=False)

        mtime = time.time()
        c.execute("UPDATE meta SET mtime = ?", (mtime,))

        conn.commit()

        logging.info("Optimizing DB...")

        db.vacuum(conn=conn)
        games_added = db.count_games(name, conn=conn)

    logging.info(
        "Updated '%s' in %.2f sec. Size: %.2f MB, Games: %i",
        filename,
        mtime - stime,
        os.path.getsize(filename) / 1024**2,
        games_added,
    )


def add_nledata_directory(path, name, filename=db.DB):
    with db.db(filename=filename, rw=True) as conn:

        logging.info("Adding dataset '%s' ('%s') to '%s' " % (name, path, filename))
        root = os.path.abspath(path)
        stime = time.time()

        c = conn.cursor()

        db.create_dataset(name, root, conn=c, commit=False)

        for xlogfile in sorted(glob.iglob(path + "/*/*.xlogfile")):

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

            gameids = db.get_most_recent_games(c.rowcount, conn=c)
            db.add_games(name, *gameids, conn=conn, commit=False)

            valid_resets = list(resets)[: len(gameids)]
            ttyrecs = [
                stem.replace("*", str(r)) for r in sorted(valid_resets, reverse=True)
            ]
            ttyrec_gen = ttyrec_data_generator(ttyrecs, gameids, root)
            c.executemany("INSERT INTO ttyrecs VALUES (?,?,?,?,?)", ttyrec_gen)

        mtime = time.time()
        c.execute("UPDATE meta SET mtime = ?", (mtime,))

        conn.commit()
        games_added = db.count_games(name, conn=conn)

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
            game_data = collections.defaultdict(lambda: -1)
            for words in line.decode("latin-1").strip().split(separator):
                key, *var = words.split("=")
                game_data[key] = "=".join(var)

            if "while" in game_data:
                game_data["death"] += " while " + game_data["while"]

            yield tuple(ctype(game_data[key]) for key, ctype in XLOGFILE_COLUMNS)
