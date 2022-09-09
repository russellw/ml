import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="build a one-file program")
parser.add_argument("src", help="source file")
parser.add_argument("--zlib", help="path/filename for z.lib")
args = parser.parse_args()

src = args.src
if not os.path.splitext(src)[1]:
    src += ".cpp"
src = os.path.join(os.path.dirname(sys.argv[0]), src)

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


cmd = "cl /MDd /Zi /I" + os.path.join(os.path.dirname(sys.argv[0])[:-3], "src")
cmd += " " + call("llvm-config --cxxflags").rstrip()
cmd += " " + src
cmd += " " + call("llvm-config --libs").rstrip()
cmd += " " + zlib
call(cmd)
