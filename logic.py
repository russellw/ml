import fractions


# distinct objects

distinct_objects = {}


class DistinctObject:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


def distinct_object(name):
    if name in distinct_objects:
        return distinct_objects[name]
    a = DistinctObject(name)
    distinct_objects[name] = a
    return a


# real number constants (must be rational, but separate type from Fraction)


class Real(fractions.Fraction):
    pass


# variables

var_name_i = 0


def reset_var_names():
    global var_name_i
    var_name_i = 0


def set_var_name(x):
    global var_name_i
    if hasattr(x, "name"):
        return
    i = var_name_i
    var_name_i += 1
    if i < 26:
        x.name = chr(65 + i)
    else:
        x.name = "Z" + str(i - 25)


class Var:
    def __init__(self, ty="individual"):
        assert ty != "bool"
        self.ty = ty

    def __repr__(self):
        if not hasattr(self, "name"):
            set_var_name(self)
        return self.name


# terms


def arithmetic_type(a):
    return typeof(a[1])


op_types = {
    "*": arithmetic_type,
    "+": arithmetic_type,
    "-": arithmetic_type,
    "/": arithmetic_type,
    "<": "bool",
    "<=": "bool",
    "=>": "bool",
    ">": "bool",
    ">=": "bool",
    "and": "bool",
    "ceil": arithmetic_type,
    "div-e": arithmetic_type,
    "div-f": arithmetic_type,
    "div-t": arithmetic_type,
    "equiv": "bool",
    "exists": "bool",
    "floor": arithmetic_type,
    "forall": "bool",
    "int?": "bool",
    "not": "bool",
    "or": "bool",
    "rat?": "bool",
    "rem-e": arithmetic_type,
    "rem-f": arithmetic_type,
    "rem-t": arithmetic_type,
    "round": arithmetic_type,
    "to-int": "int",
    "to-rat": "rat",
    "to-real": "real",
    "trunc": arithmetic_type,
    "unary-": arithmetic_type,
    "xor": "bool",
}

term_keywords = set(op_types.keys())
type_keywords = {"bool", "individual", "int", "rat", "real"}

fn_types = {}


def const(a):
    if isinstance(a, bool):
        return True
    if isinstance(a, DistinctObject):
        return True
    if isinstance(a, int):
        return True
    if isinstance(a, fractions.Fraction):
        return True


def equatable(a, b):
    ty = typeof(a)
    if ty != typeof(b):
        return
    # normally first-order logic doesn't allow equality on predicates
    # superposition calculus makes a special exception
    # for the pseudo-equation p=true
    if ty == "bool":
        return b is True
    return True


def equation(a):
    if isinstance(a, tuple) and a[0] == "=":
        return a
    return "=", a, True


def occurs(a, b, m):
    assert isinstance(a, Var)
    if a is b:
        return True
    if isinstance(b, tuple):
        for c in b:
            if occurs(a, c, m):
                return True
    if b in m:
        return occurs(a, m[b], m)


def free_vars(a):
    bound = set()
    free = []
    get_free_vars(a, bound, free)
    return free


def get_free_vars(a, bound, free):
    if isinstance(a, tuple):
        if a[0] in ("exists", "forall"):
            bound = bound.copy()
            for x in a[1]:
                bound.add(x)
            get_free_vars(a[2], bound, free)
            return
        for b in a[1:]:
            get_free_vars(b, bound, free)
        return
    if isinstance(a, Var):
        if a not in bound and a not in free:
            free.append(a)
        return


def get_fn(a, arity, m):
    if a in term_keywords:
        return
    if a not in m:
        m[a] = arity
        return
    if m[a] != arity:
        m[a] = -1


def get_fns(a, m):
    if isinstance(a, tuple):
        for b in a[1:]:
            get_fns(b, m)
        get_fn(a[0], len(a) - 1, m)
        return
    if isinstance(a, str):
        get_fn(a, 0, m)


def get_fns_clause(c, m):
    for a in c.neg:
        get_fns(a, m)
    for a in c.pos:
        get_fns(a, m)


def get_fns_clauses(cs):
    m = {}
    for c in cs:
        get_fns_clause(c, m)
    return m


def isomorphic(a, b, m):
    if typeof(a) != typeof(b):
        return
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return
        for i in range(len(a)):
            if not isomorphic(a[i], b[i], m):
                return
        return True
    if a == b:
        return True
    if isinstance(a, Var) and isinstance(b, Var):
        if a in m and b in m:
            return m[a] is m[b]
        if a not in m and b not in m:
            m[a] = b
            m[b] = a
            return True
        return


def match(a, b, m):
    if typeof(a) != typeof(b):
        return
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return
        for i in range(len(a)):
            if not match(a[i], b[i], m):
                return
        return True
    if a == b:
        return True
    if isinstance(a, Var):
        if a in m:
            return m[a] == b
        m[a] = b
        return True


def quantify(a):
    v = free_vars(a)
    if v:
        return "forall", v, a
    return a


def splice(a, path, b, i=0):
    if i == len(path):
        return b
    a = list(a)
    j = path[i]
    a[j] = splice(a[j], path, b, i + 1)
    return tuple(a)


def subst(a, m):
    if a in m:
        return subst(m[a], m)
    if isinstance(a, tuple):
        r = []
        for b in a:
            r.append(subst(b, m))
        return tuple(r)
    return a


def term_size(a):
    if isinstance(a, tuple):
        n = 0
        for b in a:
            n += term_size(b)
        return n
    return 1


def typecheck(a, ty):
    if isinstance(a, tuple):
        a = a[0]
    if a in fn_types:
        assert fn_types[a] == ty
        return
    fn_types[a] = ty


def typeof(a):
    if isinstance(a, tuple):
        o = a[0]
        if o in op_types:
            t = op_types[o]
            if isinstance(t, str):
                return t
            return t(a)
        if o in fn_types:
            return fn_types[o]
        return "individual"
    if isinstance(a, str):
        if a in fn_types:
            return fn_types[a]
        return "individual"
    if isinstance(a, Var):
        return a.ty
    if isinstance(a, DistinctObject):
        return "individual"
    if isinstance(a, Real):
        return "real"
    if isinstance(a, fractions.Fraction):
        return "rat"
    return type(a).__name__


def unequal(a, b):
    if const(a) and const(b):
        return a != b


def unify(a, b, m):
    if typeof(a) != typeof(b):
        return
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return
        for i in range(len(a)):
            if not unify(a[i], b[i], m):
                return
        return True
    if a == b:
        return True
    if isinstance(a, Var):
        return unify_var(a, b, m)
    if isinstance(b, Var):
        return unify_var(b, a, m)


def unify_var(a, b, m):
    if a in m:
        return unify(m[a], b, m)
    if b in m:
        return unify(a, m[b], m)
    if occurs(a, b, m):
        return
    m[a] = b
    return True


def unquantify(a):
    if isinstance(a, tuple) and a[0] == "forall":
        return a[2]
    return a


# simplify

number_types = {"int", "rat", "real"}


def simplify_add(a):
    x = a[1]
    y = a[2]
    ty = typeof(x)
    if ty != typeof(y) or ty not in number_types:
        raise ValueError(
            str(x) + ":" + str(typeof(x)) + " vs " + str(y) + ":" + str(typeof(y))
        )
    if const(x) and const(y):
        r = x + y
        if ty == "real":
            r = Real(r)
        return r
    return a


def simplify_sub(a):
    x = a[1]
    y = a[2]
    ty = typeof(x)
    if ty != typeof(y) or ty not in number_types:
        raise ValueError(
            str(x) + ":" + str(typeof(x)) + " vs " + str(y) + ":" + str(typeof(y))
        )
    if const(x) and const(y):
        r = x - y
        if ty == "real":
            r = Real(r)
        return r
    return a


def simplify_mul(a):
    x = a[1]
    y = a[2]
    ty = typeof(x)
    if ty != typeof(y) or ty not in number_types:
        raise ValueError(
            str(x) + ":" + str(typeof(x)) + " vs " + str(y) + ":" + str(typeof(y))
        )
    if const(x) and const(y):
        r = x * y
        if ty == "real":
            r = Real(r)
        return r
    return a


def simplify_eq(a):
    x = a[1]
    y = a[2]
    if typeof(x) != typeof(y):
        raise ValueError(
            str(x) + ":" + str(typeof(x)) + "=" + str(y) + ":" + str(typeof(y))
        )
    if not equatable(x, y):
        raise ValueError(str(x) + "=" + str(y))
    if x == y:
        return True
    if unequal(x, y):
        return False
    if y is True:
        return x
    return a


def simplify_ge(a):
    x = a[1]
    y = a[2]
    ty = typeof(x)
    if ty != typeof(y) or ty not in number_types:
        raise ValueError(
            str(x) + ":" + str(typeof(x)) + "=" + str(y) + ":" + str(typeof(y))
        )
    if const(x) and const(y):
        return x >= y
    return a


def simplify_gt(a):
    x = a[1]
    y = a[2]
    ty = typeof(x)
    if ty != typeof(y) or ty not in number_types:
        raise ValueError(
            str(x) + ":" + str(typeof(x)) + "=" + str(y) + ":" + str(typeof(y))
        )
    if const(x) and const(y):
        return x > y
    return a


def simplify_le(a):
    x = a[1]
    y = a[2]
    ty = typeof(x)
    if ty != typeof(y) or ty not in number_types:
        raise ValueError(
            str(x) + ":" + str(typeof(x)) + "=" + str(y) + ":" + str(typeof(y))
        )
    if const(x) and const(y):
        return x <= y
    return a


def simplify_lt(a):
    x = a[1]
    y = a[2]
    ty = typeof(x)
    if ty != typeof(y) or ty not in number_types:
        raise ValueError(
            str(x) + ":" + str(typeof(x)) + "=" + str(y) + ":" + str(typeof(y))
        )
    if const(x) and const(y):
        return x < y
    return a


op_simplify = {
    "+": simplify_add,
    "-": simplify_sub,
    "*": simplify_mul,
    "=": simplify_eq,
    ">": simplify_gt,
    ">=": simplify_ge,
    "<": simplify_lt,
    "<=": simplify_le,
}


def simplify(a):
    if isinstance(a, tuple):
        o = a[0]
        r = [o]
        for b in a[1:]:
            r.append(simplify(b))
        a = tuple(r)
        if o in op_simplify:
            return op_simplify[o](a)
    return a


# formulas


class Formula:
    def __init__(self, name, term, *parents):
        set_formula_name(self, name)
        self.parents = parents
        self.term = term


formula_name_i = 0
formulas_visited = set()


def reset_formula_names():
    global formula_name_i
    formula_name_i = 0


def set_formula_name(c, name):
    global formula_name_i
    if name is None:
        name = formula_name_i
        formula_name_i += 1
    elif isinstance(name, int):
        formula_name_i = max(formula_name_i, name + 1)
    c.name = name


def walk_proof(c, f):
    formulas_visited.clear()
    walk_proof1(c, f)


def walk_proof1(c, f):
    if c in formulas_visited:
        return
    formulas_visited.add(c)
    for d in c.parents:
        walk_proof1(d, f)
    f(c)


def proof_len(c):
    n = 0
    if not c:
        return 0

    def f(c):
        nonlocal n
        n += 1

    walk_proof(c, f)
    return n


# clauses


class Clause:
    def __init__(self, name, neg, pos, *parents):
        self.parents = parents

        # check structure
        for a in neg:
            etc.check_tuples(a)
        for a in pos:
            etc.check_tuples(a)

        # check types
        for a in neg:
            typecheck(a, "bool")
        for a in pos:
            typecheck(a, "bool")

        # simplify and eliminate redundancy
        r = []
        for a in neg:
            a = simplify(a)
            if a is not True:
                r.append(a)
        neg = r

        r = []
        for a in pos:
            a = simplify(a)
            if a is not False:
                r.append(a)
        pos = r

        # check structure
        for a in neg:
            etc.check_tuples(a)
        for a in pos:
            etc.check_tuples(a)

        # tautology?
        if False in neg:
            set_true(self)
            return
        if True in pos:
            set_true(self)
            return
        for a in neg:
            if a in pos:
                set_true(self)
                return

        # ok
        set_formula_name(self, name)
        self.neg = tuple(neg)
        self.pos = tuple(pos)

    def __lt__(self, other):
        return clause_size(self) < clause_size(other)

    def __repr__(self):
        return str(self.neg) + "=>" + str(self.pos)


def term_contains_arithmetic(a):
    if isinstance(a, tuple):
        for b in a:
            if term_contains_arithmetic(b):
                return True
    return typeof(a) in number_types


def clause_contains_arithmetic(c):
    for a in c.neg:
        if term_contains_arithmetic(a):
            return True
    for a in c.pos:
        if term_contains_arithmetic(a):
            return True


def clauses_contain_arithmetic(cs):
    for c in cs:
        if clause_contains_arithmetic(c):
            return True


def clause_size(c):
    return term_size(c.neg) + term_size(c.pos)


def is_false(c):
    return not c.neg and not c.pos


def is_true(c):
    return not c.neg and c.pos == (True,)


def rename_vars(c):
    m = {}

    neg = []
    for a in c.neg:
        neg.append(rename_vars1(a, m))

    pos = []
    for a in c.pos:
        pos.append(rename_vars1(a, m))

    c = Clause("renamed", neg, pos, c)
    c.renamed = True
    return c


def rename_vars1(a, m):
    if isinstance(a, tuple):
        r = []
        for b in a:
            r.append(rename_vars1(b, m))
        return tuple(r)
    if isinstance(a, Var):
        if a in m:
            return m[a]
        b = Var(a.ty)
        m[a] = b
        return b
    return a


def set_true(c):
    c.name = "tautology"
    c.neg = ()
    c.pos = (True,)


def transform_clause(c, f):
    neg = []
    for a in c.neg:
        neg.append(f(a))
    pos = []
    for a in c.pos:
        pos.append(f(a))
    return Clause(c.name, neg, pos)


def transform_clauses(cs, f):
    cs1 = []
    for c in cs:
        cs1.append(transform_clause(c, f))
    return cs1


# problems


class Problem:
    def __init__(self, name):
        self.clauses = []
        self.expected = None
        self.formulas = []
        self.name = name

    def __str__(self):
        return self.name


# test

if __name__ == "__main__":
    # test isomorphic
    r = Var("real")
    x = Var()
    y = Var()

    # atoms, equal
    m = {}
    assert isomorphic("a", "a", m)
    assert len(m) == 0

    # atoms, unequal
    m = {}
    assert not isomorphic("a", "b", m)

    # variables, equal
    m = {}
    assert isomorphic(x, x, m)
    assert len(m) == 0

    # variables, match
    m = {}
    assert isomorphic(x, y, m)
    assert len(m) == 2

    # variables, different types
    m = {}
    assert not isomorphic(x, r, m)

    # compound, equal
    m = {}
    assert isomorphic(("=", "a", "a"), ("=", "a", "a"), m)
    assert len(m) == 0

    # compound, equal
    m = {}
    assert isomorphic(("=", x, x), ("=", x, x), m)
    assert len(m) == 0

    # compound, equal
    m = {}
    assert isomorphic(("=", "a", ("f", x)), ("=", "a", ("f", x)), m)
    assert len(m) == 0

    # compound, unequal
    m = {}
    assert not isomorphic(("=", "a", "a"), ("=", "a", "b"), m)

    # compound, unequal
    m = {}
    assert not isomorphic(("=", "a", "a"), ("=", "a", x), m)

    # compound, match
    m = {}
    assert isomorphic(("=", "a", ("f", x)), ("=", "a", ("f", y)), m)
    assert len(m) == 2

    ########################################

    # https:#en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
    z = Var()

    # Succeeds. (tautology)
    m = {}
    assert unify("a", "a", m)
    assert len(m) == 0

    # a and b do not match
    m = {}
    assert not unify("a", "b", m)

    # Succeeds. (tautology)
    m = {}
    assert unify(x, x, m)
    assert len(m) == 0

    # x is unified with the constant a
    m = {}
    assert unify("a", x, m)
    assert len(m) == 1
    assert subst(x, m) == "a"

    # x and y are aliased
    m = {}
    assert unify(x, y, m)
    assert len(m) == 1
    assert subst(x, m) == subst(y, m)

    # function and constant symbols match, x is unified with the constant b
    m = {}
    assert unify(("f", "a", x), ("f", "a", "b"), m)
    assert len(m) == 1
    assert subst(x, m) == "b"

    # f and g do not match
    m = {}
    assert not unify(("f", "a"), ("g", "a"), m)

    # x and y are aliased
    m = {}
    assert unify(("f", x), ("f", y), m)
    assert len(m) == 1
    assert subst(x, m) == subst(y, m)

    # f and g do not match
    m = {}
    assert not unify(("f", x), ("g", y), m)

    # Fails. The f function symbols have different arity
    m = {}
    assert not unify(("f", x), ("f", y, z), m)

    # Unifies y with the term g(x)
    m = {}
    assert unify(("f", ("g", x)), ("f", y), m)
    assert len(m) == 1
    assert subst(y, m) == ("g", x)

    # Unifies x with constant a, and y with the term g(a)
    m = {}
    assert unify(("f", ("g", x), x), ("f", y, "a"), m)
    assert len(m) == 2
    assert subst(x, m) == "a"
    assert subst(y, m) == ("g", "a")

    # Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
    m = {}
    assert not unify(x, ("f", x), m)

    # Both x and y are unified with the constant a
    m = {}
    assert unify(x, y, m)
    assert unify(y, "a", m)
    assert len(m) == 2
    assert subst(x, m) == "a"
    assert subst(y, m) == "a"

    # As above (order of equations in set doesn't matter)
    m = {}
    assert unify("a", y, m)
    assert unify(x, y, m)
    assert len(m) == 2
    assert subst(x, m) == "a"
    assert subst(y, m) == "a"

    # Fails. a and b do not match, so x can't be unified with both
    m = {}
    assert unify(x, "a", m)
    assert len(m) == 1
    assert not unify("b", x, m)

    ########################################

    # match is a subset of unify, matching variables only on the left
    # gives different results in several cases
    # in particular, has no notion of an occurs check
    # assumes the inputs have disjoint variables

    # Succeeds. (tautology)
    m = {}
    assert match("a", "a", m)
    assert len(m) == 0

    # a and b do not match
    m = {}
    assert not match("a", "b", m)

    # Succeeds. (tautology)
    m = {}
    assert match(x, x, m)
    assert len(m) == 0

    # x is unified with the constant a
    m = {}
    assert not match("a", x, m)

    # x and y are aliased
    m = {}
    assert match(x, y, m)
    assert len(m) == 1
    assert m[x] == y

    # function and constant symbols match, x is unified with the constant b
    m = {}
    assert match(("f", "a", x), ("f", "a", "b"), m)
    assert len(m) == 1
    assert m[x] == "b"

    # f and g do not match
    m = {}
    assert not match(("f", "a"), ("g", "a"), m)

    # x and y are aliased
    m = {}
    assert match(("f", x), ("f", y), m)
    assert len(m) == 1
    assert m[x] == y

    # f and g do not match
    m = {}
    assert not match(("f", x), ("g", y), m)

    # Fails. The f function symbols have different arity
    m = {}
    assert not match(("f", x), ("g", y, z), m)

    # Unifies y with the term g(x)
    m = {}
    assert not match(("f", ("g", x)), ("f", y), m)

    # Unifies x with constant a, and y with the term g(a)
    m = {}
    assert not match(("f", ("g", x), x), ("f", y, "a"), m)

    # Both x and y are unified with the constant a
    m = {}
    assert match(x, y, m)
    assert match(y, "a", m)
    assert len(m) == 2
    assert m[x] == y
    assert m[y] == "a"

    # As above (order of equations in set doesn't matter)
    m = {}
    assert not match("a", y, m)

    # Fails. a and b do not match, so x can't be unified with both
    m = {}
    assert match(x, "a", m)
    assert len(m) == 1
    assert not match("b", x, m)

    ########################################

    print("ok")
