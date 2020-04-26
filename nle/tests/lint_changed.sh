#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This shell script lints only the things that changed in the most recent change.
# It also ignores deleted files, so that black and flake8 don't explode.

set -e

CMD=""
while getopts ":bfa" opt; do
    case ${opt} in
        b )
            CMD="black"
            ;;
        f )
            CMD="flake8"
            ;;
        a )
            ALLCMD=1
            ;;
        \? ) echo "Usage: cmd [-a] [-b]"
             exit 1
             ;;
    esac
done

# HACK change this when circleci implements target branch envvar
MASTER_BRANCH="master"
CHANGED_FILES=`git diff --diff-filter=d --name-only $MASTER_BRANCH...`


if [[ "$CHANGED_FILES" == "" ]]; then
    echo "No changed files."
    exit 0
else
    echo "--- Changed files:"
    echo "$CHANGED_FILES"
    echo "---"
fi

if [[ $ALLCMD -eq 1 || $CMD == "black" ]]; then
    PY_CHANGED_FILES=`echo "$CHANGED_FILES" | grep '\.py$' | grep -v -E '^scripts\/plotting\/.*$' | tr '\n' ' '`
    echo "--- Changed files (python only):"
    echo "$PY_CHANGED_FILES"
    echo "---"
    echo "Running black tests..."

    command -v black >/dev/null || \
        ( echo "Please install black." && false )
    # only output if something needs to change
    black --check $PY_CHANGED_FILES
fi

if [[ $ALLCMD -eq 1 || $CMD == "flake8" ]]; then
    PY_CHANGED_FILES=`echo "$CHANGED_FILES" | grep '\.py$' | grep -v -E '^scripts\/plotting\/.*$' | tr '\n' ' '`
    echo "--- Changed files (python only):"
    echo "$PY_CHANGED_FILES"
    echo "---"
    echo "Running flake8 tests..."

    # flake8 3.7+ required for --select
    # soft complaint on too-long-lines
    flake8 --select=E501 --show-source $PY_CHANGED_FILES
    # hard complaint on really long lines
    flake8 --max-line-length=127 --show-source $PY_CHANGED_FILES
fi
