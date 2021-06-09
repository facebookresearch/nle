#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
from nle.nethack import actions
from nle.scripts import ttyplay

if __name__ == "__main__":
    ttyplay.INPUTS = actions._ACTIONS_DICT
    ttyplay.main()
