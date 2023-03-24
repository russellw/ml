def is_var(a):
    return isinstance(a, str) and a.startswith("$")


def occurs(d, a, b):
    assert a.startswith("$")
    if a == b:
        return True
    if isinstance(b, tuple):
        for bi in b:
            if occurs(d, a, bi):
                return True
    if b in d:
        return occurs(d, a, d[b])


def unify_var(d, a, b):
    assert is_var(a)
    if a in d:
        return unify(d, d[a], b)
    if b in d:
        return unify(d, a, d[b])
    if occurs(d, a, b):
        return
    d[a] = b
    return True


def unify(d, a, b):
    # this version of unify skips the type check
    # because it makes no sense to ask the type of a type
    if a == b:
        return True

    if is_var(a):
        return unify_var(d, a, b)
    if is_var(b):
        return unify_var(d, b, a)

    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        for i in range(len(a)):
            if not unify(d, a[i], b[i]):
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
    assert unify(d, a, a)
    assert len(d) == 0

    # a and b do not match
    d = {}
    assert not unify(d, a, b)

    # Succeeds. (tautology)
    d = {}
    assert unify(d, x, x)
    assert len(d) == 0

    # x is unified with the constant a
    d = {}
    assert unify(d, a, x)
    assert len(d) == 1
    assert replace(x, d) == a

    # x and y are aliased
    d = {}
    assert unify(d, x, y)
    assert len(d) == 1
    assert replace(x, d) == replace(y, d)

    # function and constant symbols match, x is unified with the constant b
    d = {}
    assert unify(d, (f, a, x), (f, a, b))
    assert len(d) == 1
    assert replace(x, d) == b

    # f and g do not match
    d = {}
    assert not unify(d, (f, a), (g, a))

    # x and y are aliased
    d = {}
    assert unify(d, (f, x), (f, y))
    assert len(d) == 1
    assert replace(x, d) == replace(y, d)

    # f and g do not match
    d = {}
    assert not unify(d, (f, x), (g, y))

    # Fails. The f function symbols have different arity
    d = {}
    assert not unify(d, (f, x), (f, y, z))

    # Unifies y with the term g(x)
    d = {}
    assert unify(d, (f, (g, x)), (f, y))
    assert len(d) == 1
    assert replace(y, d) == (g, x)

    # Unifies x with constant a, and y with the term g(a)
    d = {}
    assert unify(d, (f, (g, x), x), (f, y, a))
    assert len(d) == 2
    assert replace(x, d) == a
    assert replace(y, d) == (g, a)

    # Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
    d = {}
    assert not unify(d, x, (f, x))

    # Both x and y are unified with the constant a
    d = {}
    assert unify(d, x, y)
    assert unify(d, y, a)
    assert len(d) == 2
    assert replace(x, d) == a
    assert replace(y, d) == a

    # As above (order of equations in set doesn't matter)
    d = {}
    assert unify(d, a, y)
    assert unify(d, x, y)
    assert len(d) == 2
    assert replace(x, d) == a
    assert replace(y, d) == a

    # Fails. a and b do not match, so x can't be unified with both
    d = {}
    assert unify(d, x, a)
    assert len(d) == 1
    assert not unify(d, b, x)

    print("ok")
