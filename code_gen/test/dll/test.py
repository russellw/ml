import os
import subprocess

test_dir = os.path.dirname(os.path.realpath(__file__))


def cc(s):
    subprocess.check_call(("clang", "-S", "-emit-llvm", s))


for root, dirs, files in os.walk(test_dir):
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext in (".c", ".cpp"):
            f = os.path.join(root, f)
            cc(f)
with open("modules.tsv", "w") as f:
    f.write("poly.dll\tsquare.ll\n")
    f.write("poly.dll\tcube.ll\n")
    f.write("\tmain.ll\n")
subprocess.check_call(("clang", "a.ll"))
subprocess.check_call(("a"))
