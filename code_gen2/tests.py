import os
import subprocess


here = os.path.dirname(os.path.realpath(__file__))


for root, dirs, files in os.walk(here):
    for f in files:
        if f == "test.py":
            f = os.path.join(root, f)
            subprocess.check_call(("python", f))
