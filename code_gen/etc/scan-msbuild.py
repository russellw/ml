import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("src")
args = parser.parse_args()


def cl(s):
    print(str(line) + "\tCL" + s, end="")


lines = open(os.path.join(args.src, "msbuild.log")).readlines()


def link(dll, ext):
    print(str(line) + "\tLINK" + dll)


line = 0
for s in lines:
    line += 1

    m = re.match(r".*\Wtracker\.exe", s, re.IGNORECASE)
    if m:
        continue

    m = re.match(r".*\Wcl\.exe", s, re.IGNORECASE)
    if m:
        cl(s[len(m[0]) :])
        continue

    m = re.match(r"(.*\Wlink\.exe).*/IMPLIB:(\S+).*/DLL", s)
    if m:
        link(s[len(m[0]) :], ".dll")
        continue

    m = re.match(r"(.*\Wlink\.exe).*/IMPLIB:(\S+)", s)
    if m:
        link(s[len(m[0]) :], ".exe")
        continue
