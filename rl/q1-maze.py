import random

import numpy as np
import torch

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


episodes = 1
for episode in range(episodes):
    state = start
    while maze[state] == 0:
        print(state)
        a = select_action(state)
        s = next(state, a)
        state = s
