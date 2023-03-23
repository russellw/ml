import sys

import dimacs


def replace(clauses, key, val):
    ok = False
    r = []
    for neg, pos in clauses:
        if key in neg or key in pos:
            ok = True
        neg = tuple([val if a == key else a for a in neg])
        pos = tuple([val if a == key else a for a in pos])
        if False in neg or True in pos:
            continue
        neg = tuple([a for a in neg if a is not True])
        pos = tuple([a for a in pos if a is not False])
        r.append((neg, pos))
    assert ok
    return tuple(r)


def dpll(strategy, clauses):
    if ((), ()) in clauses:
        return False
    if clauses == ():
        return True

    for neg, pos in clauses:
        if len(neg) == 1 and not pos:
            return dpll(strategy, replace(clauses, neg[0], True))
        if not neg and len(pos) == 1:
            return dpll(strategy, replace(clauses, pos[0], True))

    key, val = strategy(clauses)
    assert isinstance(key, str)
    assert isinstance(val, bool)
    return dpll(strategy, replace(clauses, key, val)) or dpll(
        strategy, replace(clauses, key, not val)
    )


def default_strategy(clauses):
    neg, pos = clauses[0]
    return (neg + pos)[0], False


if __name__ == "__main__":
    clauses = dimacs.read_file(sys.argv[1])
    print(dpll(default_strategy, clauses))
