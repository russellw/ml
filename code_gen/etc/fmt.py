import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description="format source code")
parser.add_argument("-i", action="store_true", help="inplace edit")
parser.add_argument("files", nargs="*")
args = parser.parse_args()

if not args.i:
    print("-i not specified, taking no action")
    exit(1)

for a in args.files:
    for root, dirs, files in os.walk(a):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in (".c", ".cc", ".cpp", ".h"):
                f = os.path.join(root, f)
                subprocess.check_call(("clang-format", "-i", "-style=file", f))
