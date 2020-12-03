import fractions

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

# subsumption
p = fn("p")
p.ty = "bool"
p1 = fn("p1")
p1.ty = "bool", "individual"
p2 = fn("p2")
p2.ty = "bool", "individual", "individual"

q = fn("q")
q.ty = "bool"
q1 = fn("q1")
q1.ty = "bool", "individual"
q2 = fn("q2")
q2.ty = "bool", "individual", "individual"

# false <= false
c = Clause(None, [], [])
d = Clause(None, [], [])
assert subsumes(c, d)

# false < p
c = Clause(None, [], [])
d = Clause(None, [], [p])
assert subsumes(c, d)
assert not subsumes(d, c)

# p <= p
c = Clause(None, [], [p])
d = Clause(None, [], [p])
assert subsumes(c, d)

# !p <= !p
c = Clause(None, [p], [])
d = Clause(None, [p], [])
assert subsumes(c, d)

# p < p | p
c = Clause(None, [], [p])
d = Clause(None, [], [p, p])
assert subsumes(c, d)
assert not subsumes(d, c)

# p !<= !p
c = Clause(None, [], [p])
d = Clause(None, [p], [])
assert not subsumes(c, d)
assert not subsumes(d, c)

# p | q <= q | p
c = Clause(None, [], [p, q])
d = Clause(None, [], [q, p])
assert subsumes(c, d)
assert subsumes(d, c)

# p | q < p | q | p
c = Clause(None, [], [p, q])
d = Clause(None, [], [p, q, p])
assert subsumes(c, d)
assert not subsumes(d, c)

# p(a) | p(b) | q(a) | q(b) | <= p(a) | q(a) | p(b) | q(b)
c = Clause(None, [], [(p1, a), (p1, b), (q1, a), (q1, b)])
d = Clause(None, [], [(p1, a), (q1, a), (p1, b), (q1, b)])
assert subsumes(c, d)
assert subsumes(d, c)

# p(x,y) < p(a,b)
c = Clause(None, [], [(p2, x, y)])
d = Clause(None, [], [(p2, a, b)])
assert subsumes(c, d)
assert not subsumes(d, c)

# p(x,x) !<= p(a,b)
c = Clause(None, [], [(p2, x, x)])
d = Clause(None, [], [(p2, a, b)])
assert not subsumes(c, d)
assert not subsumes(d, c)

# p(x) <= p(y)
c = Clause(None, [], [(p1, x)])
d = Clause(None, [], [(p1, y)])
assert subsumes(c, d)
assert subsumes(d, c)

# p(x) | p(a(x)) | p(a(a(x))) <= p(y) | p(a(y)) | p(a(a(y)))
c = Clause(None, [], [(p1, x), (p1, (f, x)), (p1, (f, (f, x)))])
d = Clause(None, [], [(p1, y), (p1, (f, y)), (p1, (f, (f, y)))])
assert subsumes(c, d)
assert subsumes(d, c)

# p(x) | p(a) < p(a) | p(b)
c = Clause(None, [], [(p1, x), (p1, a)])
d = Clause(None, [], [(p1, a), (p1, b)])
assert subsumes(c, d)
assert not subsumes(d, c)

# p(x) | p(a(x)) <= p(a(y)) | p(y)
c = Clause(None, [], [(p1, x), (p1, (f, x))])
d = Clause(None, [], [(p1, (f, y)), (p1, y)])
assert subsumes(c, d)
assert subsumes(d, c)

# p(x) | p(a(x)) | p(a(a(x))) <= p(a(a(y))) | p(a(y)) | p(y)
c = Clause(None, [], [(p1, x), (p1, (f, x)), (p1, (f, (f, x)))])
d = Clause(None, [], [(p1, (f, (f, y))), (p1, (f, y)), (p1, y)])
assert subsumes(c, d)
assert subsumes(d, c)

# (a = x) < (a = b)
c = Clause(None, [], [("=", a, x)])
d = Clause(None, [], [("=", a, b)])
assert subsumes(c, d)
assert not subsumes(d, c)

# (x = a) < (a = b)
c = Clause(None, [], [("=", x, a)])
d = Clause(None, [], [("=", a, b)])
assert subsumes(c, d)
assert not subsumes(d, c)

# !p(y) | !p(x) | q(x) < !p(a) | !p(b) | q(b)
c = Clause(None, [(p1, y), (p1, x)], [(q1, x)])
d = Clause(None, [(p1, a), (p1, b)], [(q1, b)])
assert subsumes(c, d)
assert not subsumes(d, c)

# !p(x) | !p(y) | q(x) < !p(a) | !p(b) | q(b)
c = Clause(None, [(p1, x), (p1, y)], [(q1, x)])
d = Clause(None, [(p1, a), (p1, b)], [(q1, b)])
assert subsumes(c, d)
assert not subsumes(d, c)

# p(x,a(x)) !<= p(a(y),a(y))
c = Clause(None, [], [(p1, x, (f, x))])
d = Clause(None, [], [(p1, (f, y), (f, y))])
assert not subsumes(c, d)
assert not subsumes(d, c)

########################################

assert typeof(fractions.Fraction("1/3")) == "rat"
assert typeof(Real("1/3")) == "real"

# simplify
assert simplify(("+", 1, 2)) == 3
assert simplify(
    ("+", fractions.Fraction("1/3"), fractions.Fraction("1/3"))
) == fractions.Fraction("2/3")
assert simplify(("+", Real("1/3"), Real("1/3"))) == Real("2/3")

########################################

print("ok")
