#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.

files="server.js config.js app/dashboard.html app/actions.js app/third_party/ttyplay.js"
for file in ${files}; do
    echo "\nCheking ${file}..."
    ./node_modules/.bin/eslint ${file} --fix
done
