#!/usr/bin/env bash

set -e
set -o

git clean -dfX

ENVPATH=`python -m sysconfig | grep -e "^[[:space:]]*base =" | cut -d'"' -f 2`
rm -rfv $ENVPATH/lib/games/nethack
rm -v $ENVPATH/bin/nethack

pushd sys/unix

sh setup.sh hints/macosx

popd

PREFIX=$ENVPATH make -j
PREFIX=$ENVPATH make install
