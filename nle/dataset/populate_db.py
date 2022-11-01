import collections
import datetime
import glob
import os
import re
import time
from functools import partial

from nle import dataset as nld

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
ALT_TIMEFMT = re.compile(r"(.*\.\d\d)_(\d\d)_(\d\d.*)")


def altorg_filename_to_timestamp(filename):
    ts = filename.split("/")[-1][:-11]
    # We accept time format HH_MM_SS or HH:MM:SS, but convert for ISO format.
    ts = ALT_TIMEFMT.sub(r"\1:\2:\3", ts)
    try:
        ts = datetime.datetime.fromisoformat(ts)
    except AttributeError:
        # Python 3.6 doesnt have fromisoformat
        this_date, this_time = ts.split(".")
        this_date = [int(x) for x in this_date.split("-")]
        this_time = [int(x) for x in this_time.split(":")]
        this_datetime = this_date + this_time
        ts = datetime.datetime(*this_datetime)
    except ValueError:
        print("Skipping: '%s'" % filename)
        return -1
    return ts.replace(tzinfo=datetime.timezone.utc).timestamp()


def assign_ttyrecs_to_games(ttyrecs, games):
    """Algorithm to assign a players ttyrecs to their games, knowing that only one game
    can be played at any one time."""
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

            # Design choice: We will add the ttyrec with a -ve gameid
            # so that it will not be picked up when selecting by dataset="altorg"
            # but will still exist in the database
            assigned[tt][2] = -games[gg][0]
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


def add_altorg_directory(path, name, filename=nld.db.DB):
    """This function can be used to add the `altorg` dataset to a database.

    Once the altorg dataset has been downloaded, this function will parse its
    contents and create the dataset.

    The altorg directory structure should look like:

        altorg/
        ├── user1/
        │   ├── 2019-07-25.22:03:29.ttyrec.bz2
        │   └── 2019-07-30.16:06:23.ttyrec.bz2
        ├── user2/
        │   ├── 2019-07-25.22:03:29.ttyrec.bz2
        │   └── 2019-07-30.16:06:23.ttyrec.bz2
        ...
        ├── xlogfile.1
        ├── xlogfile.2
        ...
        ├── about.txt
        └── blacklist.txt

    Note that unlike `nle` ttyrecs, altorg episodes may be split into parts.
    We use a simple algorithm based on the file creation times and xlogfile times
    for the start and end of games to try to assign ttyrecs to games, knowing
    a player can only ever be playing on game one altorg at a time.

    This algorithm should be deterministic and always return the same dataset
    from an empty database, regardless of environment.

    """

    with nld.db.db(filename=filename, rw=True) as conn:
        print("Adding dataset '%s' ('%s') to '%s' " % (name, path, filename))

        root = os.path.abspath(path)
        stime = time.time()
        c = conn.cursor()

        # 1. Check if the dataset name exists, and add the root.
        # NB: alt.org ttyrecs are version 1, and have suffix "ttyrec.bz2"
        nld.db.create_dataset(name, root, ttyrec_version=1, conn=c, commit=False)

        # 2. Add games from xlogfile to `games` table, then `datasets` table.
        for xlogfile in reversed(
            sorted(glob.iglob(str(os.path.join(path, "xlogfile.*"))))
        ):
            sep = ":" if xlogfile.endswith(".txt") else "\t"
            game_gen = game_data_generator(xlogfile, separator=sep)
            insert_sql = f"""
                INSERT INTO games
                VALUES (NULL, {','.join('?' for _ in XLOGFILE_COLUMNS)} )
            """
            c.executemany(insert_sql, game_gen)

            gameids = nld.db.get_most_recent_games(c.rowcount, conn=c)
            nld.db.add_games(name, *gameids, conn=c, commit=False)
            print("Found %i games in '%s'" % (len(gameids), xlogfile))

        # 3. Find all the (unblacklisted) ttyrecs belonging to each player
        #    and all the games belonging to each player.
        with open(os.path.join(path, "blacklist.txt"), "r") as f:
            blacklisted_ttyrecs = set(str(os.path.join(path, p)) for p in f.readlines())

        ttyrecs_dict = collections.defaultdict(list)
        for ttyrec in glob.iglob(path + "/*/*.ttyrec.bz2"):
            if ttyrec in blacklisted_ttyrecs:
                continue
            ttyrecs_dict[ttyrec.split("/")[-2].lower()].append(ttyrec)

        games_dict = collections.defaultdict(list)
        for pname, gameid, start, end in nld.db.get_games(name, conn=c):
            games_dict[pname.lower()].append((gameid, start, end))

        # 4. Attempt assign each player's ttyrecs to each player's games.
        #    If successful, insert into `ttyrecs` table
        print("Matching up ttyrecs to games...")
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

        pseudonyms = []
        for i, pname in enumerate(games_dict):
            for gameid, _, _ in games_dict[pname]:
                pseudonyms.append((f"Player{i}", gameid, pname))
        pseudnymize = "UPDATE games SET name=? WHERE gameid=? AND lower(name)=?"
        c.executemany(pseudnymize, pseudonyms)

        # 5. Purge 'empty' games from `datasets` and `games` table.
        games_with_ttyrecs = """SELECT DISTINCT(gameid) FROM ttyrecs"""
        nld.db.delete_games_with_select(
            games_with_ttyrecs, not_in=True, conn=conn, commit=False
        )

        # 6. Purge short games where the player quit almost immediately.
        start_scummed_games = """
           SELECT gameid FROM games
           WHERE (turns <=10 AND (death = "escaped" OR death ="quit")) OR turns<=0"""
        nld.db.delete_games_with_select(start_scummed_games, conn=conn, commit=False)

        mtime = time.time()
        c.execute("UPDATE meta SET mtime = ?", (mtime,))

        # 7. Commit and wrap up (optimize the db).
        conn.commit()
        print("Optimizing DB...")
        nld.db.vacuum(conn=conn)
        games_added = nld.db.count_games(name, conn=conn)

    print(
        "Updated '%s' in %.2f sec. Size: %.2f MB, Games: %i"
        % (filename, mtime - stime, os.path.getsize(filename) / 1024**2, games_added)
    )


def add_nledata_directory(path, name, filename=nld.db.DB):
    """This function can be used to add any `nle_data` dataset to a database.

    Full games that are generated by an env such as:

        `gym.make("NetHackChallenge-v0", savedir="", save_ttyrec_every=k)`

    come with the following structure:

    The directory structure should look like:

        nle_data/
        ├── 20220414-112633_sdfd83ns/
        │   ├── nle.3968599.0.ttyrec.bz2
        │   ├── nle.3968599.k.ttyrec.bz2
        │   └── nle.3968599.xlogfile
        └── <date>-<time>_<random>/
            ├── nle.<process-id>.0.ttyrec.bz2
            ├── nle.<process-id>.k.ttyrec.bz2
            └── nle.<process-id>.xlogfile

    This algorithm should be deterministic and always return the same dataset
    from an empty database, regardless of environment.
    """
    with nld.db.db(filename=filename, rw=True) as conn:
        print("Adding dataset '%s' ('%s') to '%s' " % (name, path, filename))

        root = os.path.abspath(path)
        stime = time.time()
        c = conn.cursor()

        # 1. Check if the dataset name exists, and add the root.
        nld.db.create_dataset(name, root, conn=c, commit=False)

        # 2. For each xlogfile, read the games and take only the games that
        #   correspond to the ttyrecs that exist in the enclosing directory.
        for xlogfile in sorted(glob.iglob(path + "/*/*.xlogfile")):
            stem = xlogfile.replace(".xlogfile", ".*.ttyrec*.bz2")

            files = set(glob.iglob(stem))
            ttyrecnames = set(f.split("/")[-1] for f in files)
            versions = set(f.split("ttyrec")[-1].replace(".bz2", "") for f in files)
            assert len(versions) == 1, "Cannot add ttyrecs with different versions"
            version = versions.pop()

            if version == "":
                raise AssertionError(
                    "Ttyrec version (* in ttyrec*.bz2) must be > 1 for NLE data."
                )

            c.execute(
                "UPDATE roots SET ttyrec_version = ? WHERE dataset_name = ?",
                (int(version), name),
            )

            ttyrecs = []
            ttydir = str(os.path.dirname(xlogfile))

            _filter = partial(
                xlogfile_gen_filter,
                ttyrecs=ttyrecs,
                ttyrecnames=ttyrecnames,
                ttydir=ttydir,
            )

            # 3. Add games to `games` and `datasets` table.
            game_gen = game_data_generator(xlogfile, filter=_filter)
            insert_sql = f"""
                INSERT INTO games
                VALUES (NULL, {','.join('?' for _ in XLOGFILE_COLUMNS)} )
            """
            c.executemany(insert_sql, game_gen)
            gameids = nld.db.get_most_recent_games(c.rowcount, conn=c)
            nld.db.add_games(name, *gameids, conn=conn, commit=False)

            # 4. Add ttyrecs to `ttyrecs` table.
            # Note gameids are "most recently added" so must be reversed.
            ttyrec_gen = ttyrec_data_generator(ttyrecs, reversed(gameids), root)
            c.executemany("INSERT INTO ttyrecs VALUES (?,?,?,?,?)", ttyrec_gen)

        mtime = time.time()
        c.execute("UPDATE meta SET mtime = ?", (mtime,))

        conn.commit()
        nld.db.vacuum(conn=conn)
        games_added = nld.db.count_games(name, conn=conn)

    print(
        "Updated '%s' in %.2f sec. Size: %.2f MB, Games: %i"
        % (filename, mtime - stime, os.path.getsize(filename) / 1024**2, games_added)
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


def xlogfile_gen_filter(gen, ttyrecnames, ttyrecs, ttydir):
    """Filter lines of the xlogile, keeping files in `ttyrecnames` and storing the
    the accepted paths in `ttyrecs`."""
    # The `xlogfile` may have more rows than files in directory
    # due to 'save_ttyrec_every' option in env.py, so filter these out.
    # If we do find a file, we will save it to be added later.
    for line in gen:
        ttyrecname = line.decode("ascii").split("ttyrecname=")[-1].strip()
        if ttyrecname in ttyrecnames:
            ttyrecs.append(ttydir + "/" + ttyrecname)
            yield line
