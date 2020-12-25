import subprocess
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

logger = logging.getLogger()
logger.addHandler(
    logging.FileHandler(datetime.datetime.now().strftime("logs/0E %Y-%m-%d %H%M%S.log"))
)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)
pr_buf = ""


def pr(a):
    global pr_buf
    pr_buf += str(a)


def prn(a=""):
    global pr_buf
    logger.info(pr_buf + str(a))
    pr_buf = ""


def debug(a):
    logger.debug(str(a), stack_info=True)


prn(sys.argv)
prn()


def do_file(filename):
    global attempted
    global solved
    filename = filename.replace("\\", "/")
    if filename.startswith("C:/ml/"):
        filename = filename[6:]

    # list file
    if os.path.splitext(filename)[1] == ".lst":
        for s in open(filename):
            do_file(s.strip())
        return

    prn(filename)
    attempted += 1
    try:
        p = subprocess.run(
            ["bin/eprover", "--auto", "-p", filename],
            capture_output=True,
            encoding="utf-8",
            timeout=3,
            check=True,
        )
        prn(p.stdout)
        solved += 1
    except subprocess.TimeoutExpired:
        prn("Timeout")
        prn()
    except subprocess.CalledProcessError as p:
        prn(p.stdout)
        if p.returncode == 9 and "GaveUp" in p.stdout:
            pass
        else:
            prn("Error")
            prn(p.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run E prover")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    start = time.time()
    os.environ["TPTP"] = "TPTP"
    attempted = 0
    solved = 0
    for filename in args.files:
        if os.path.isfile(filename):
            do_file(filename)
            continue
        for root, dirs, files in os.walk(filename):
            for fname in files:
                do_file(os.path.join(root, fname))
    prn(f"solved {solved}/{attempted} = {solved*100/attempted}%")
    prn(f"{time.time() - start:.3f} seconds")
