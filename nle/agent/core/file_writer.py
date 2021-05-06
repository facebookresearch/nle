# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import csv
import datetime
import json
import logging
import os
import time
import weakref


def _save_metadata(path, metadata):
    metadata["date_save"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)


def gather_metadata():
    metadata = dict(
        date_start=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        env=os.environ.copy(),
        successful=False,
    )

    # Git metadata.
    try:
        import git
    except ImportError:
        logging.warning(
            "Couldn't import gitpython module; install it with `pip install gitpython`."
        )
    else:
        try:
            repo = git.Repo(search_parent_directories=True)
            metadata["git"] = {
                "commit": repo.commit().hexsha,
                "is_dirty": repo.is_dirty(),
                "path": repo.git_dir,
            }
            if not repo.head.is_detached:
                metadata["git"]["branch"] = repo.active_branch.name
        except git.InvalidGitRepositoryError:
            pass

    if "git" not in metadata:
        logging.warning("Couldn't determine git data.")

    # Slurm metadata.
    if "SLURM_JOB_ID" in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
        metadata["slurm"] = {}
        for k in slurm_env_keys:
            d_key = k.replace("SLURM_", "").replace("SLURMD_", "").lower()
            metadata["slurm"][d_key] = os.environ[k]

    return metadata


class FileWriter:
    def __init__(self, xp_args=None, rootdir="~/palaas"):
        if rootdir == "~/palaas":
            # make unique id in case someone uses the default rootdir
            xpid = "{proc}_{unixtime}".format(
                proc=os.getpid(), unixtime=int(time.time())
            )
            rootdir = os.path.join(rootdir, xpid)
        self.basepath = os.path.expandvars(os.path.expanduser(rootdir))

        self._tick = 0

        # metadata gathering
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # we need to copy the args, otherwise when we close the file writer
        # (and rewrite the args) we might have non-serializable objects (or
        # other nasty stuff).
        self.metadata["args"] = copy.deepcopy(xp_args)

        formatter = logging.Formatter("%(message)s")
        self._logger = logging.getLogger("palaas/out")

        # to stdout handler
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        self._logger.addHandler(shandle)
        self._logger.setLevel(logging.INFO)

        # to file handler
        if not os.path.exists(self.basepath):
            self._logger.info("Creating log directory: %s", self.basepath)
            os.makedirs(self.basepath, exist_ok=True)
        else:
            self._logger.info("Found log directory: %s", self.basepath)

        self.paths = dict(
            msg="{base}/out.log".format(base=self.basepath),
            logs="{base}/logs.csv".format(base=self.basepath),
            fields="{base}/fields.csv".format(base=self.basepath),
            meta="{base}/meta.json".format(base=self.basepath),
        )

        self._logger.info("Saving arguments to %s", self.paths["meta"])
        if os.path.exists(self.paths["meta"]):
            self._logger.warning(
                "Path to meta file already exists. " "Not overriding meta."
            )
        else:
            self.save_metadata()

        self._logger.info("Saving messages to %s", self.paths["msg"])
        if os.path.exists(self.paths["msg"]):
            self._logger.warning(
                "Path to message file already exists. " "New data will be appended."
            )

        fhandle = logging.FileHandler(self.paths["msg"])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        self._logger.info("Saving logs data to %s", self.paths["logs"])
        self._logger.info("Saving logs' fields to %s", self.paths["fields"])
        self.fieldnames = ["_tick", "_time"]
        if os.path.exists(self.paths["logs"]):
            self._logger.warning(
                "Path to log file already exists. " "New data will be appended."
            )
            # Override default fieldnames.
            with open(self.paths["fields"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                if len(lines) > 0:
                    self.fieldnames = lines[-1]
            # Override default tick: use the last tick from the logs file plus 1.
            with open(self.paths["logs"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                # Need at least two lines in order to read the last tick:
                # the first is the csv header and the second is the first line
                # of data.
                if len(lines) > 1:
                    self._tick = int(lines[-1][0]) + 1

        self._fieldfile = open(self.paths["fields"], "a")
        self._fieldwriter = csv.writer(self._fieldfile)
        self._fieldfile.flush()
        self._logfile = open(self.paths["logs"], "a")
        self._logwriter = csv.DictWriter(self._logfile, fieldnames=self.fieldnames)

        # Auto-close (and save) on destruction.
        weakref.finalize(self, _save_metadata, self.paths["meta"], self.metadata)

    def log(self, to_log, tick=None, verbose=False):
        if tick is not None:
            raise NotImplementedError
        else:
            to_log["_tick"] = self._tick
            self._tick += 1
        to_log["_time"] = time.time()

        old_len = len(self.fieldnames)
        for k in to_log:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        if old_len != len(self.fieldnames):
            self._fieldwriter.writerow(self.fieldnames)
            self._fieldfile.flush()
            self._logger.info("Updated log fields: %s", self.fieldnames)

        if to_log["_tick"] == 0:
            self._logfile.write("# %s\n" % ",".join(self.fieldnames))

        if verbose:
            self._logger.info(
                "LOG | %s",
                ", ".join(["{}: {}".format(k, to_log[k]) for k in sorted(to_log)]),
            )

        self._logwriter.writerow(to_log)
        self._logfile.flush()

    def close(self, successful=True):
        self.metadata["successful"] = successful
        self.save_metadata()

        for f in [self._logfile, self._fieldfile]:
            f.close()

    def save_metadata(self):
        _save_metadata(self.paths["meta"], self.metadata)
