import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from life import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--density", help="density of random grid", type=float, default=0.5
)
parser.add_argument("-r", "--rand", help="random pattern", action="store_true")
parser.add_argument("-s", "--seed", help="random number seed", type=int)
parser.add_argument("-z", "--size", help="grid size", type=int, default=256)
parser.add_argument("file", nargs="?")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

size = args.size

g = None
if args.rand is not None:
    g = randgrid(size, args.density)
if args.file:
    g = read(args.file)


def update(frame):
    g.run()
    data = g.get_data(0, 0, size, size)
    img = ax.imshow(data, interpolation="nearest")
    return img


fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update, interval=0)
plt.show()
