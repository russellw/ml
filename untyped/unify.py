def is_var(a):
    return isinstance(a, str) and a.startswith("$")


def occurs(a, b, d):
    assert a.startswith("$")
    if a == b:
        return True
    if isinstance(b, tuple):
        for bi in b:
            if occurs(a, bi, d):
                return True
    if b in d:
        return occurs(a, d[b], d)


def unify_var(a, b, d):
    assert is_var(a)
    if a in d:
        return unify(d[a], b, d)
    if b in d:
        return unify(a, d[b], d)
    if occurs(a, b, d):
        return
    d[a] = b
    return True


def unify(a, b, d):
    # this version of unify skips the type check
    # because it makes no sense to ask the type of a type
    if a == b:
        return True

    if is_var(a):
        return unify_var(a, b, d)
    if is_var(b):
        return unify_var(b, a, d)

    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        for i in range(len(a)):
            if not unify(a[i], b[i], d):
                return
        return True


def replace(a, d):
    if isinstance(a, tuple):
        return tuple([replace(b, d) for b in a])
    if a in d:
        return replace(d[a], d)
    return a


if __name__ == "__main__":
    a = "a"
    b = "b"
    f = "f"
    g = "g"

    # unification here works on type variables instead of first-order logic variables,
    # but the algorithm is close enough to identical, that first-order test cases can be reused
    x = "$x"
    y = "$y"
    z = "$z"

    # https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms

    # Succeeds. (tautology)
    d = {}
    assert unify(a, a, d)
    assert len(d) == 0

    # a and b do not match
    d = {}
    assert not unify(a, b, d)

    # Succeeds. (tautology)
    d = {}
    assert unify(x, x, d)
    assert len(d) == 0

    # x is unified with the constant a
    d = {}
    assert unify(a, x, d)
    assert len(d) == 1
    assert replace(x, d) == a

    # x and y are aliased
    d = {}
    assert unify(x, y, d)
    assert len(d) == 1
    assert replace(x, d) == replace(y, d)

    # function and constant symbols match, x is unified with the constant b
    d = {}
    assert unify((f, a, x), (f, a, b), d)
    assert len(d) == 1
    assert replace(x, d) == b

    # f and g do not match
    d = {}
    assert not unify((f, a), (g, a), d)

    # x and y are aliased
    d = {}
    assert unify((f, x), (f, y), d)
    assert len(d) == 1
    assert replace(x, d) == replace(y, d)

    # f and g do not match
    d = {}
    assert not unify((f, x), (g, y), d)

    # Fails. The f function symbols have different arity
    d = {}
    assert not unify((f, x), (f, y, z), d)

    # Unifies y with the term g(x)
    d = {}
    assert unify((f, (g, x)), (f, y), d)
    assert len(d) == 1
    assert replace(y, d) == (g, x)

    # Unifies x with constant a, and y with the term g(a)
    d = {}
    assert unify((f, (g, x), x), (f, y, a), d)
    assert len(d) == 2
    assert replace(x, d) == a
    assert replace(y, d) == (g, a)

    # Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
    d = {}
    assert not unify(x, (f, x), d)

    # Both x and y are unified with the constant a
    d = {}
    assert unify(x, y, d)
    assert unify(y, a, d)
    assert len(d) == 2
    assert replace(x, d) == a
    assert replace(y, d) == a

    # As above (order of equations in set doesn't matter)
    d = {}
    assert unify(a, y, d)
    assert unify(x, y, d)
    assert len(d) == 2
    assert replace(x, d) == a
    assert replace(y, d) == a

    # Fails. a and b do not match, so x can't be unified with both
    d = {}
    assert unify(x, a, d)
    assert len(d) == 1
    assert not unify(b, x, d)

    print("ok")
