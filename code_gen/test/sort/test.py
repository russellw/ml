import os
import subprocess

test_dir = os.path.dirname(os.path.realpath(__file__))


def cc(s):
    subprocess.check_call("clang -D_CRT_SECURE_NO_DEPRECATE -S -emit-llvm " + s)


for root, dirs, files in os.walk(test_dir):
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext in (".c", ".cpp"):
            f = os.path.join(root, f)
            cc(f)
subprocess.check_call(
    "olivine --main=sort.exe " + os.path.join(test_dir, "modules.tsv")
)
