# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import pprint
import sys

Entry = collections.namedtuple("Entry", "size hash line file")


def main():
    allocs = {}
    with open(sys.argv[1]) as heaplog:
        for line in heaplog:
            entries = line.split()
            if entries[0] == "+":
                entry = Entry(*entries[1:])
                allocs[entry.hash] = entry
            else:
                entry = Entry(None, *entries[1:])
                if entry.hash not in allocs:
                    print("dealloc not found in allocs:", line)
                    continue
                del allocs[entry.hash]

    pprint.pprint(allocs)


if __name__ == "__main__":
    main()
