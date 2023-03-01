import argparse
import random

from torch import nn
from torch.utils.data import Dataset, DataLoader

import zz

exts = set()
exts.add(".java")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r", "--scramble", help="amount of scrambling", type=int, default=10
)
parser.add_argument("-s", "--seed", help="random number seed")
parser.add_argument("-z", "--size", help="chunk size", type=int, default=100)
parser.add_argument("src_dir")
args = parser.parse_args()

if args.seed is not None:
    random.seed(options.seed)

v = zz.get_chunks(exts, args.src_dir, args.size)[0]
print(v)
print(zz.scramble(v, args.scramble))
