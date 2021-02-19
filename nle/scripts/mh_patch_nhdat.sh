#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.

ORIGDIR=$(pwd)

VARDIR=$1
HACKDIR=$2
LIBDIR=$3

if [ -z "$4" ]
then
    des_file="$ORIGDIR/mylevel.des"
else
    des_file=$4
fi

if [ ! -d $VARDIR/lib ]
then
    cp -r $LIBDIR $VARDIR
fi

cd $VARDIR/lib

cp $des_file mylevel.des
$HACKDIR/lev_comp mylevel.des
rm -rf mylevel.des

$HACKDIR/dlb cf nhdat *
mv nhdat $VARDIR

cd $ORIGDIR