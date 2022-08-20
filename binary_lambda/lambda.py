class Closure:
    def __init__(self, env, body):
        self.env = env
        self.body = body

    def __call__(self, x):
        return ev((x,) + self.env, self.body)


I = ("L", 0)


def ev(env, a):
    if isinstance(a, int):
        return env[a]
    if a[0] == "A":
        return ev(env, a[1])(ev(env, a[2]))
    if a[0] == "L":
        return Closure(env, a[1])
    raise ValueError(a)


if __name__ == "__main__":
    print(ev((), ("A", I, I)))
    assert ev((), ("A", I, I)) == I
