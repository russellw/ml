class Env(dict):
    def __init__(self, outer=None, params=(), args=()):
        self.outer = outer
        self.update(zip(params, args))

    def get(self, k):
        env = self
        while env:
            if k in env:
                return env[k]
            env = env.outer
        raise ValueError(k)

    def keys1(self):
        s = set()
        env = self
        while env:
            s.update(env.keys())
            env = env.outer
        return s
