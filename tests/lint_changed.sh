#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This shell script lints only the things that changed in the most recent change.
# It also ignores deleted files, so that black and flake8 don't explode.

set -e

CMD=""
while getopts ":bfca" opt; do
    case ${opt} in
        b )
            CMD="black"
            ;;
        f )
            CMD="flake8"
            ;;
        c )
            CMD="clang-format"
            ;;
        a )
            ALLCMD=1
            ;;
        \? ) echo "Usage: cmd [-a] [-b] [-c]"
             exit 1
             ;;
    esac
done

MASTER_BRANCH="rl"
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
    PY_CHANGED_FILES=`echo "$CHANGED_FILES" | grep '\.py$' | tr '\n' ' '`
    if [[ "$PY_CHANGED_FILES" == "" ]]; then
        echo "No python files changed."
    else
        echo "--- Changed files (python only):"
        echo "$PY_CHANGED_FILES"
        echo "---"
        echo "Running black tests..."

        command -v black >/dev/null || \
            ( echo "Please install black." && false )
        # only output if something needs to change
        black --check $PY_CHANGED_FILES
    fi
fi

if [[ $ALLCMD -eq 1 || $CMD == "flake8" ]]; then
    PY_CHANGED_FILES=`echo "$CHANGED_FILES" | grep '\.py$' | tr '\n' ' '`
    if [[ "$PY_CHANGED_FILES" == "" ]]; then
        echo "No python files changed."
    else
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
fi

if [[ $ALLCMD -eq 1 || $CMD == "clang-format" ]]; then
    CC_CHANGED_FILES=`echo "$CHANGED_FILES" | grep -E '^win\/rl\/\w*\.(cc|c|cpp|h|hpp)$' | tr '\n' ' '`
    if [[ "$CC_CHANGED_FILES" == "" ]]; then
        echo "No cpp files changed."
    else
        echo "--- Changed files (cpp only):"
        echo "$CC_CHANGED_FILES"
        echo "---"
        echo "Running clang-format tests..."

        ./tests/run-clang-format.py $(echo $CC_CHANGED_FILES)
    fi
fi
