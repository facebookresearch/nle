#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.

ORIGDIR=$(pwd)

VARDIR=$1
HACKDIR=$2
LIBDIR=$3
DESFILE=$4

if [ ! -d $VARDIR/lib ]
then
    cp -r $LIBDIR $VARDIR
fi

cd $VARDIR/lib

cp $DESFILE mylevel.des
$HACKDIR/lev_comp mylevel.des
rm -rf mylevel.des

$HACKDIR/dlb cf nhdat *
mv nhdat $VARDIR

cd $ORIGDIR
