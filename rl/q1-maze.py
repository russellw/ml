import argparse
import random

import numpy as np
import torch

# command line
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random number seed", type=int)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

size = 8
start = 0, 0
pit = size // 2, size // 2
goal = size - 1, size - 1

maze = torch.zeros(size, size, dtype=torch.int8)
for i in range(size * size // 10):
    maze[random.randrange(size), random.randrange(size)] = 9
maze[start] = 0
maze[pit] = -1
maze[goal] = 1
print(maze)

actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def next(s, a):
    i, j = s
    di, dj = a

    i += di
    if not 0 <= i < size:
        return s

    j += dj
    if not 0 <= j < size:
        return s

    if maze[i, j] == 9:
        return s
    return i, j


def successors(s):
    return [next(s, a) for a in actions]


def reward(s):
    assert maze[s] != 9
    return maze[s]


Q = torch.zeros(size, size)


def estimated_reward(s):
    assert maze[s] != 9
    if maze[s]:
        return maze[s]
    return Q[s]


def select_action(s):
    e = 0.1
    if random.random() < e:
        return random.choice(actions)
    v = np.array([estimated_reward(next(s, a)) for a in actions])
    return actions[np.random.choice(np.flatnonzero(v == v.max()))]


discount = 0.9


def update_Q(s):
    Q[s] += discount * torch.tensor([estimated_reward(t) for t in successors(s)]).max()


for episode in range(3):
    state = start
    while maze[state] == 0:
        update_Q(state)
        a = select_action(state)
        state = next(state, a)
    print(Q)
