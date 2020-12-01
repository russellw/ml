from logic import *

# test distinct objects
assert distinct_object("a") == distinct_object("a")
assert distinct_object("a") != distinct_object("b")
assert not (distinct_object("a") != distinct_object("a"))
assert not (distinct_object("a") == distinct_object("b"))

########################################

# test isomorphic
a = fn("a")
a.ty = "individual"
b = fn("b")
b.ty = "individual"
f = fn("f")
f.ty = "individual", "individual"
f2 = fn("f2")
f2.ty = "individual", "individual", "individual"
g = fn("g")
g.ty = "individual", "individual"
g2 = fn("g2")
g2.ty = "individual", "individual", "individual"
r = Var("real")
x = Var("individual")
y = Var("individual")

# atoms, equal
m = {}
assert isomorphic(a, a, m)
assert len(m) == 0

# atoms, unequal
m = {}
assert not isomorphic(a, b, m)

# variables, equal
m = {}
assert isomorphic(x, x, m)
assert len(m) == 0

# variables, match
m = {}
assert isomorphic(x, y, m)
assert len(m) == 2


# compound, equal
m = {}
assert isomorphic(("=", a, a), ("=", a, a), m)
assert len(m) == 0

# compound, equal
m = {}
assert isomorphic(("=", x, x), ("=", x, x), m)
assert len(m) == 0

# compound, equal
m = {}
assert isomorphic(("=", a, (f, x)), ("=", a, (f, x)), m)
assert len(m) == 0

# compound, unequal
m = {}
assert not isomorphic(("=", a, a), ("=", a, b), m)

# compound, unequal
m = {}
assert not isomorphic(("=", a, a), ("=", a, x), m)

# compound, match
m = {}
assert isomorphic(("=", a, (f, x)), ("=", a, (f, y)), m)
assert len(m) == 2

########################################

# https:#en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
z = Var("individual")

# Succeeds. (tautology)
m = {}
assert unify(a, a, m)
assert len(m) == 0

# a and b do not match
m = {}
assert not unify(a, b, m)

# Succeeds. (tautology)
m = {}
assert unify(x, x, m)
assert len(m) == 0

# x is unified with the constant a
m = {}
assert unify(a, x, m)
assert len(m) == 1
assert subst(x, m) == a

# x and y are aliased
m = {}
assert unify(x, y, m)
assert len(m) == 1
assert subst(x, m) == subst(y, m)

# function and constant symbols match, x is unified with the constant b
m = {}
assert unify((f2, a, x), (f2, a, b), m)
assert len(m) == 1
assert subst(x, m) == b

# f and g do not match
m = {}
assert not unify((f, a), (g, a), m)

# x and y are aliased
m = {}
assert unify((f, x), (f, y), m)
assert len(m) == 1
assert subst(x, m) == subst(y, m)

# f and g do not match
m = {}
assert not unify((f, x), (g, y), m)

# Fails. The f function symbols have different arity
m = {}
assert not unify((f, x), (f2, y, z), m)

# Unifies y with the term g(x)
m = {}
assert unify((f, (g, x)), (f, y), m)
assert len(m) == 1
assert subst(y, m) == (g, x)

# Unifies x with constant a, and y with the term g(a)
m = {}
assert unify((f2, (g, x), x), (f2, y, a), m)
assert len(m) == 2
assert subst(x, m) == a
assert subst(y, m) == (g, a)

# Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
m = {}
assert not unify(x, (f, x), m)

# Both x and y are unified with the constant a
m = {}
assert unify(x, y, m)
assert unify(y, a, m)
assert len(m) == 2
assert subst(x, m) == a
assert subst(y, m) == a

# As above (order of equations in set doesn't matter)
m = {}
assert unify(a, y, m)
assert unify(x, y, m)
assert len(m) == 2
assert subst(x, m) == a
assert subst(y, m) == a

# Fails. a and b do not match, so x can't be unified with both
m = {}
assert unify(x, a, m)
assert len(m) == 1
assert not unify(b, x, m)

########################################

# match is a subset of unify, matching variables only on the left
# gives different results in several cases
# in particular, has no notion of an occurs check
# assumes the inputs have disjoint variables

# Succeeds. (tautology)
m = {}
assert match(a, a, m)
assert len(m) == 0

# a and b do not match
m = {}
assert not match(a, b, m)

# Succeeds. (tautology)
m = {}
assert match(x, x, m)
assert len(m) == 0

# x is unified with the constant a
m = {}
assert not match(a, x, m)

# x and y are aliased
m = {}
assert match(x, y, m)
assert len(m) == 1
assert m[x] == y

# function and constant symbols match, x is unified with the constant b
m = {}
assert match((f2, a, x), (f2, a, b), m)
assert len(m) == 1
assert m[x] == b

# f and g do not match
m = {}
assert not match((f, a), (g, a), m)

# x and y are aliased
m = {}
assert match((f, x), (f, y), m)
assert len(m) == 1
assert m[x] == y

# f and g do not match
m = {}
assert not match((f, x), (g, y), m)

# Fails. The f function symbols have different arity
m = {}
assert not match((f, x), (g2, y, z), m)

# Unifies y with the term g(x)
m = {}
assert not match((f, (g, x)), (f, y), m)

# Unifies x with constant a, and y with the term g(a)
m = {}
assert not match((f2, (g, x), x), (f2, y, a), m)

# Both x and y are unified with the constant a
m = {}
assert match(x, y, m)
assert match(y, a, m)
assert len(m) == 2
assert m[x] == y
assert m[y] == a

# As above (order of equations in set doesn't matter)
m = {}
assert not match(a, y, m)

# Fails. a and b do not match, so x can't be unified with both
m = {}
assert match(x, a, m)
assert len(m) == 1
assert not match(b, x, m)

########################################

print("ok")
