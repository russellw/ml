import argparse
import datetime
import fractions
import heapq
import inspect
import itertools
import logging
import os
import re
import sys
import time


def do_file(filename):
    if "^" in filename:
        return
    print(filename + "\t", end="")
    s = open(filename).read()
    print(str(len(s)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for filename in args.files:
        if os.path.isfile(filename):
            do_file(filename)
            continue
        for root, dirs, files in os.walk(filename):
            for fname in files:
                do_file(os.path.join(root, fname))
