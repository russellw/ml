import argparse
import random
import statistics

import pandas as pd

# Decision trees
def op_lt(a, m):
    x = evaluate(a[1], m)
    y = evaluate(a[2], m)
    return x < y


ops = {"<": op_lt}


def evaluate(a, m):
    if isinstance(a, tuple):
        return ops[a[0]](a, m)
    if isinstance(a, str):
        return m[a]
    return a


class Tree:
    def __init__(self, test, children=None):
        self.test = test
        if not children:
            children = []
        self.children = children

    def evaluate(self, m):
        test = self.test
        if test is True:
            return self.result
        r = evaluate(test, m)
        return self.children[int(r)].evaluate(m)


def unpack(df):
    # separate the output column
    y_name = df.columns[-1]
    y_df = df[y_name]
    df = df.drop(y_name, axis=1)

    rs = []
    rows = df.shape[0]
    for i in range(rows):
        m = {}
        for j in range(len(df.columns)):
            m[df.columns[j]] = df.iat[i, j]
        rs.append((m, y_df[i]))
    return rs


# args
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="CSV file")
args = parser.parse_args()


def argmin(f, xs):
    xmin = xs[0]
    ymin = f(xmin)
    for x in xs[1:]:
        y = f(x)
        if y < ymin:
            ymin = y
            xmin = x
    return xmin, ymin


# data
df = pd.read_csv(args.filename)
print(df)
print()

rs = unpack(df)
print(rs[0])

features = list(rs[0][0].keys())


def separate(rs, k, t):
    lo = []
    hi = []
    for m, y in rs:
        if m[k] < t:
            lo.append((m, y))
        else:
            hi.append((m, y))
    return lo, hi


def error_rs(rs):
    if not rs:
        return 0.0
    mean = statistics.mean(map(lambda r: r[1], rs))
    e = 0.0
    for m, y in rs:
        e += (y - mean) ** 2
    return e


def tree(rs):
    # error for a given choice of feature
    t = 0

    def error_k(k):
        nonlocal t

        # what values do we have for this feature?
        xs = []
        for m, y in rs:
            x = m[k]
            if isinstance(x, str):
                print(x)
                return 1e100
            xs.append(x)
        xs.sort()

        # candidate thresholds between values
        ts = []
        for i in range(len(xs) - 1):
            ts.append(xs[i] + xs[i + 1] / 2)

        # error for a given choice of threshold
        def error_t(t):
            lo, hi = separate(rs, k, t)
            return error_rs(lo) * len(lo) + error_rs(hi) * len(hi)

        # best threshold
        t, e = argmin(error_t, ts)
        return e

    k, e = argmin(error_k, features)
    lo, hi = separate(rs, k, t)
    limit = 10
    if len(lo) < limit or len(hi) < limit:
        T = Tree(True)
        T.result = statistics.mean(map(lambda r: r[1], rs))
        return T
    test = "<", k, t
    T = Tree(test)
    T.children.append(tree(lo))
    T.children.append(tree(hi))
    return T


def split_train_test(rs):
    random.shuffle(rs)
    n = len(rs)
    i = n // 4
    return rs[i:], rs[:i]


train, test = split_train_test(rs)
T = tree(train)
e = 0.0
for m, y in test:
    e += (T.evaluate(m) - y) ** 2
print("mean squared error:", e / len(test))
