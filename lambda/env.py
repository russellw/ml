class Env(dict):
    def __init__(self, outer, params, args):
        self.outer = outer
        self.update(zip(params, args))

    def get(self, k):
        env = self
        while env:
            if k in env:
                return env[k]
            env = env.outer
        raise Exception(k)
