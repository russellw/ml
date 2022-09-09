import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="build Olivine")
parser.add_argument("--zlib", help="path/filename for z.lib")
args = parser.parse_args()

src = os.path.join(os.path.dirname(sys.argv[0]), "*.cpp")

zlib = r"C:\lib\z.lib"
if args.zlib:
    zlib = args.zlib


def call(cmd):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    stdout = str(stdout, "utf-8")
    if p.returncode:
        stderr = str(stderr, "utf-8")
        print(stdout, end="")
        print(stderr, end="")
        raise Exception(str(p.returncode))
    return stdout


cmd = "cl /Feolivine /MDd /Zi"
cmd += " " + call("llvm-config --cxxflags").rstrip()
cmd += " " + src
cmd += " " + call("llvm-config --libs").rstrip()
cmd += " " + zlib
call(cmd)
