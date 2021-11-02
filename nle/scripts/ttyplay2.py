#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
from nle.nethack import actions
from nle.scripts import ttyplay


def main():
    ttyplay.INPUTS = actions._ACTIONS_DICT
    ttyplay.main()


if __name__ == "__main__":
    main()
