import random

import numpy as np
import torch

size = 10
actions = 2


def next(s, a):
    if a == 0:
        a = -1
    s += a
    if s == size:
        s = -1
    return s


def reward(s):
    if s < 0:
        return -1
    return s / size


Q = torch.zeros(size)


def estimated_reward(s):
    if s < 0:
        return -1
    return Q[s]


def select_action(s):
    e = 0.1
    if random.random() < e:
        return random.randrange(actions)
    v = np.array([estimated_reward(next(s, a)) for a in range(actions)])
    return np.random.choice(np.flatnonzero(v == v.max()))


episodes = 1
for episode in range(episodes):
    state = 0
    while state != size - 1:
        print(state)
        a = select_action(state)
        s = next(state, a)
        if s >= 0:
            state = s
