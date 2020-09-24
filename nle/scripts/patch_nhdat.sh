#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
set -euo pipefail

ORIGDIR=$(pwd)

HACKDIR=$(python -c 'import pkg_resources; print(pkg_resources.resource_filename("nle", "nethackdir"), end="")')

TMPDIR=$(mktemp -d)

echo $TMPDIR

mkdir $TMPDIR/dat
cd $TMPDIR/dat
cp $HACKDIR/dat/dungeon.def .
patch --ignore-whitespace <<'EOF'
--- dungeon.def    2019-03-01 15:21:08.000000000 +0100
+++ dungeon.def    2020-09-23 19:17:51.000000000 +0200
@@ -15,6 +15,7 @@
 #

 DUNGEON:       "The Dungeons of Doom" "D" (25, 5)
+LEVEL:         "mylevel" "none" @ (1,1)
 ALIGNMENT:     unaligned
 BRANCH:                "The Gnomish Mines" @ (2, 3)
 LEVEL:         "rogue" "R" @ (15, 4)

EOF
$HACKDIR/makedefs -e  # Looks for ../dat/dungeon.def.
$HACKDIR/dgn_comp dungeon.pdf

cp $HACKDIR/dat/oracle.des mylevel.des
patch --ignore-whitespace <<'EOF'
--- mylevel.des    2019-03-01 15:21:08.000000000 +0100
+++ mylevel.des    2020-09-23 19:01:56.000000000 +0200
@@ -5,7 +5,7 @@
 # Oracle level
 #

-LEVEL: "oracle"
+LEVEL: "mylevel"

 ROOM: "ordinary" , lit, (3,3), (center,center), (11,9) {
   OBJECT:('`',"statue"),(0,0),montype:'C',1
EOF
$HACKDIR/lev_comp mylevel.des

mkdir $TMPDIR/contents
cd $TMPDIR/contents
$HACKDIR/dlb xf $HACKDIR/nhdat
cp -f ../dat/dungeon ../dat/mylevel.lev .
$HACKDIR/dlb cf nhdat *
cp nhdat $ORIGDIR

cd $ORIGDIR

rm -rf $TMPDIR
