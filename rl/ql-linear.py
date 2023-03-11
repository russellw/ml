import random

import torch


def clamp(lo, x, hi):
    return min(max(x, lo), hi)


size = 10
actions = 2


def select_action():
    e = 0.1
    if random.random() < e:
        return random.randrange(actions)
    return Q[state].argmax().item()


def reward(a):
    return a


Q = torch.rand(size, actions)
print(Q)


def update_state(a):
    global state
    if a == 0:
        a = -1
    state += a
    state = clamp(0, state, size - 1)


episodes = 10
turns = 10

for episode in range(episodes):
    state = 0
    for t in range(turns):
        a = select_action()
        update_state(a)
