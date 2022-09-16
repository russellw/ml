import argparse
import os
import random
import re
import subprocess
import shutil


def call(*cmd):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    if stderr:
        stderr = str(stderr, "utf-8")
        raise Exception(stderr)
    if p.returncode:
        stdout = str(stdout, "utf-8")
        print(stdout)
        raise Exception(str(p.returncode))
    return stdout


parser = argparse.ArgumentParser(description="Run test cases")
parser.add_argument("files", nargs="*")
args = parser.parse_args()

test_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.join(test_dir, "..")


def cc(f):
    call(
        "cl",
        "/DDEBUG",
        "/EHsc",
        "/I" + main_dir,
        "/WX",
        "/Zi",
        "/nologo",
        f,
        os.path.join(main_dir, "lo", "*.cc"),
        "dbghelp.lib",
    )


def do(f):
    print(f)
    shutil.copy2(f, "a.cpp")
    cc("a.cpp")
    s = call("a")
    print(s)
    open("a1.cpp", "wb").write(s)
    cc("a1.cpp")
    subprocess.check_call('a1')


tests = [test_dir]
if args.files:
    tests = args.files
for test in tests:
    if os.path.isfile(test):
        do(test)
        continue
    if not os.path.isdir(test):
        print(test + ": not found")
        exit(1)
    for root, dirs, files in os.walk(test):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in (".cc", ".cpp"):
                do(os.path.join(root, f))
