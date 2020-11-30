import argparse
import fractions
import inspect
import itertools
import os
import re
import sys
import time


def check_tuples(a):
    if isinstance(a, tuple):
        for b in a:
            check_tuples(b)
        return
    if isinstance(a, list):
        raise ValueError(a)


def debug(x):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(x)}", file=sys.stderr)


######################################## terms


class DistinctObject:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


distinct_objects = {}


def distinct_object(name):
    if name in distinct_objects:
        return distinct_objects[name]
    a = DistinctObject(name)
    distinct_objects[name] = a
    return a


# real number constants must be rational, but separate type from Fraction
class Real(fractions.Fraction):
    pass


# constants are just functions of arity zero
class Fn:
    def __init__(self, name, rty, args):
        self.name = name
        if not rty:
            return
        self.ty = rty
        if args:
            self.ty = (rty,) + tuple([typeof(a) for a in args])

    def __repr__(self):
        return self.name


fns = {}


def fn(name, rty, args):
    if name in fns:
        return fns[name]
    a = Fn(name, rty, args)
    fns[name] = a
    return a


# named types are handled as functions
types = {}


def mktype(name):
    if name in types:
        return types[name]
    a = Fn(name, None, [])
    types[name] = a
    return a


# first-order variables cannot be boolean
class Var:
    def __init__(self, ty="individual"):
        if not ty:
            return
        assert ty != "bool"
        self.ty = ty


def arity(a):
    if isinstance(a, str):
        arities = {
            "*": 2,
            "+": 2,
            "-": 2,
            "/": 2,
            "<": 2,
            "<=": 2,
            ">": 2,
            ">=": 2,
            "ceil": 1,
            "div-e": 2,
            "div-f": 2,
            "div-t": 2,
            "eqv": 2,
            "floor": 1,
            "int?": 1,
            "not": 1,
            "rat?": 1,
            "rem-e": 2,
            "rem-f": 2,
            "rem-t": 2,
            "round": 1,
            "to-int": 1,
            "to-rat": 1,
            "to-real": 1,
            "trunc": 1,
            "unary-": 1,
        }
        return arities[a]
    ty = typeof(a)
    if isinstance(ty, tuple):
        n = len(ty) - 1
        assert n
        return n
    return 0


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


def free_vars(a):
    bound = set()
    free = []

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

    get_free_vars(a, bound, free)
    return free


def imp(a, b):
    return "or", ("not", a), b


def isomorphic(a, b, m):
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
        return tuple([subst(b, m) for b in a])
    return a


def term_size(a):
    if isinstance(a, tuple):
        n = 0
        for b in a:
            n += term_size(b)
        return n
    return 1


def unequal(a, b):
    if const(a) and const(b):
        return a != b


def unify(a, b, m):
    if typeof(a) != typeof(b):
        return
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return
        if a[0] != b[0]:
            return
        for i in range(1, len(a)):
            if not unify(a[i], b[i], m):
                return
        return True
    if a == b:
        return True

    def unify_var(a, b, m):
        if a in m:
            return unify(m[a], b, m)
        if b in m:
            return unify(a, m[b], m)
        if occurs(a, b, m):
            return
        m[a] = b
        return True

    if isinstance(a, Var):
        return unify_var(a, b, m)
    if isinstance(b, Var):
        return unify_var(b, a, m)


def unquantify(a):
    if isinstance(a, tuple) and a[0] == "forall":
        return a[2]
    return a


def walk(a, f):
    if isinstance(a, tuple):
        for b in a:
            walk(b, f)
    f(a)


# types
def typeof(a):
    if isinstance(a, tuple):
        o = a[0]
        if isinstance(o, str):
            if o in (
                "exists",
                "forall",
                "eqv",
                "=",
                "<",
                "<=",
                "and",
                "or",
                "int?",
                "rat?",
            ):
                return "bool"
            if o.startswith("to-"):
                return o[3:]
            return typeof(a[1])
        ty = typeof(o)
        if not isinstance(ty, tuple):
            raise ValueError(a)
        return ty[0]
    if isinstance(a, str):
        raise ValueError(a)
    if isinstance(a, Fn) or isinstance(a, Var):
        return a.ty
    if isinstance(a, DistinctObject):
        return "individual"
    if isinstance(a, Real):
        return "real"
    if isinstance(a, fractions.Fraction):
        return "rat"
    return type(a).__name__


# first step of type inference:
# unify to figure out how all the unspecified types can be made consistent
def type_unify(wanted, a, m):
    # this version of unify skips the type check
    # because it makes no sense to ask the type of a type
    def unify(a, b, m):
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b):
                return
            for i in range(len(a)):
                if not unify(a[i], b[i], m):
                    return
            return True
        if a == b:
            return True

        def unify_var(a, b, m):
            if a in m:
                return unify(m[a], b, m)
            if b in m:
                return unify(a, m[b], m)
            if occurs(a, b, m):
                return
            m[a] = b
            return True

        if isinstance(a, Var):
            return unify_var(a, b, m)
        if isinstance(b, Var):
            return unify_var(b, a, m)

    if not unify(wanted, typeof(a), m):
        print(m)
        raise ValueError(f"{wanted} != typeof({a}): {typeof(a)}")
    if isinstance(a, tuple):
        o = a[0]

        # predefined function
        if isinstance(o, str):
            # quantifiers require body boolean
            if o in ("exists", "forall"):
                type_unify("bool", a[2], m)
                return

            # all arguments boolean
            if o in ("and", "or", "eqv", "not"):
                for i in range(1, len(a)):
                    type_unify("bool", a[i], m)
                return

            # all arguments of the same type
            actual = typeof(a[1])
            for i in range(2, len(a)):
                type_unify(actual, a[i], m)
            return

        # user-defined function
        # we already unified the return type
        # by virtue of unifying the type of the whole expression
        # so now just need to unify parameters with arguments
        ty = typeof(o)
        assert isinstance(ty, tuple)
        for i in range(1, len(a)):
            type_unify(ty[i], a[i], m)
        return


# second step of type inference:
# fill in actual types for all the type variables
def type_set(a, m):
    if isinstance(a, tuple):
        for b in a:
            type_set(b, m)
        return
    if isinstance(a, Fn) or isinstance(a, Var):
        a.ty = subst(a.ty, m)
        if isinstance(a.ty, tuple):
            r = []
            for b in a.ty:
                if isinstance(b, Var):
                    b = "individual"
                r.append(b)
            a.ty = tuple(r)
            return
        if isinstance(a.ty, Var):
            a.ty = "individual"
            return
        return


# third step of type inference:
# check the types are correct
def type_check(wanted, a):
    if wanted != typeof(a):
        raise ValueError(f"{wanted} != typeof({a})")
    if isinstance(a, tuple):
        o = a[0]

        # predefined function
        if isinstance(o, str):
            # quantifiers require body boolean
            if o in ("exists", "forall"):
                for x in a[1]:
                    if x.ty == "bool":
                        raise ValueError(a)
                type_check("bool", a[2])
                return

            # all arguments boolean
            if o in ("and", "or", "eqv", "not"):
                for i in range(1, len(a)):
                    type_check("bool", a[i])
                return

            # all arguments int
            if o in ("div-e", "div-f", "div-t", "rem-e", "rem-f", "rem-t"):
                for i in range(1, len(a)):
                    type_check("bool", a[i])
                return

            # all arguments of the same type
            actual = typeof(a[1])
            for i in range(2, len(a)):
                type_check(actual, a[i])

            # =
            if o == "=":
                return

            # numbers
            if actual not in ("int", "rat", "real"):
                raise ValueError(a)

            # rational or real
            if o == "div" and actual == "int":
                raise ValueError(a)
            return

        return
    if isinstance(a, Fn):
        if isinstance(a.ty, tuple):
            for b in a.ty[1:]:
                if b == "bool":
                    raise ValueError(a)
        return
    if isinstance(a, Var):
        if a.ty == "bool":
            raise ValueError(a)
        return


######################################## simplify


def simplify_add(a):
    x = a[1]
    y = a[2]
    ty = typeof(x)
    if ty != typeof(y) or ty not in ("int", "rat", "real"):
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
    if ty != typeof(y) or ty not in ("int", "rat", "real"):
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
    if ty != typeof(y) or ty not in ("int", "rat", "real"):
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
    if ty != typeof(y) or ty not in ("int", "rat", "real"):
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
    if ty != typeof(y) or ty not in ("int", "rat", "real"):
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
    if ty != typeof(y) or ty not in ("int", "rat", "real"):
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
    if ty != typeof(y) or ty not in ("int", "rat", "real"):
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


######################################## logic


class Formula:
    def __init__(self, name, term, *parents):
        set_formula_name(self, name)
        self.parents = parents
        self.term = term


formula_name_i = 0


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
    visited = set()

    def rec(c):
        if c in visited:
            return
        visited.add(c)
        for d in c.parents:
            rec(d)
        f(c)

    rec(c, f)


class Clause:
    def __init__(self, name, neg, pos, *parents):
        # check structure
        for a in neg:
            check_tuples(a)
        for a in pos:
            check_tuples(a)

        # ok
        set_formula_name(self, name)
        self.neg = tuple(neg)
        self.pos = tuple(pos)
        self.parents = parents

    def __lt__(self, other):
        return clause_size(self) < clause_size(other)

    def __repr__(self):
        return str(self.neg) + "=>" + str(self.pos)


def term_contains_arith(a):
    if isinstance(a, tuple):
        for b in a:
            if term_contains_arith(b):
                return True
    return typeof(a) in ("int", "rat", "real")


def clause_contains_arith(c):
    for a in c.neg:
        if term_contains_arith(a):
            return True
    for a in c.pos:
        if term_contains_arith(a):
            return True


def clauses_contain_arith(cs):
    for c in cs:
        if clause_contains_arith(c):
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


class Problem:
    def __init__(self, name):
        self.name = name
        self.formulas = []
        self.clauses = []
        self.expected = None


######################################## TPTP


defined_fns = {
    "$ceiling": "ceil",
    "$difference": "-",
    "$floor": "floor",
    "$greater": ">",
    "$greatereq": ">=",
    "$is_int": "int?",
    "$is_rat": "rat?",
    "$less": "<",
    "$lesseq": "<=",
    "$product": "*",
    "$quotient": "/",
    "$quotient_e": "div-e",
    "$quotient_f": "div-f",
    "$quotient_t": "div-t",
    "$remainder_e": "rem-e",
    "$remainder_f": "rem-f",
    "$remainder_t": "rem-t",
    "$round": "round",
    "$sum": "+",
    "$to_int": "to-int",
    "$to_rat": "to-rat",
    "$to_real": "to-real",
    "$truncate": "trunc",
    "$uminus": "unary-",
}

defined_types = {
    "$o": "bool",
    "$i": "individual",
    "$int": "int",
    "$rat": "rat",
    "$real": "real",
}


def szs_success(s):
    if s in ("ContradictoryAxioms", "Theorem", "Unsatisfiable"):
        return True
    if s in ("CounterSatisfiable", "Satisfiable"):
        return True


def bool_szs(s):
    if s in ("ContradictoryAxioms", "Theorem", "Unsatisfiable"):
        return False
    if s in ("CounterSatisfiable", "Satisfiable"):
        return True
    raise ValueError(s)


def unquote(s):
    if s[0] in ("'", '"'):
        return s[1:-1]
    return s


def weird(s):
    if not s[0].islower():
        return True
    for c in s:
        if not c.isalnum() and c != "_":
            return True
    if s in term_keywords:
        return True


def weird_type(s):
    if not s[0].islower():
        return True
    for c in s:
        if not c.isalnum() and c != "_":
            return True
    if s in type_keywords:
        return True


# parser
class Inappropriate(Exception):
    pass


def read_tptp1(filename, select=True):
    global expected
    text = open(filename).read()
    if text and text[-1] != "\n":
        text += "\n"

    # tokenizer
    ti = 0
    tok = ""

    def err(msg):
        line = 1
        for i in range(ti):
            if text[i] == "\n":
                line += 1
        raise ValueError(f"{filename}:{line}: {repr(tok)}: {msg}")

    def lex():
        global expected
        nonlocal ti
        nonlocal tok
        while ti < len(text):
            c = text[ti]

            # space
            if c.isspace():
                ti += 1
                continue

            # line comment
            if c == "%":
                i = ti
                while text[ti] != "\n":
                    ti += 1
                if problem.expected is None:
                    m = re.match(r"%\s*Status\s*:\s*(\w+)", text[i:ti])
                    if m:
                        problem.expected = m[1]
                continue

            # block comment
            if text[ti : ti + 2] == "/*":
                ti += 2
                while text[ti : ti + 2] != "*/":
                    ti += 1
                ti += 2
                continue

            # word
            if c.isalpha() or c == "$":
                i = ti
                ti += 1
                while text[ti].isalnum() or text[ti] == "_":
                    ti += 1
                tok = text[i:ti]
                return

            # quote
            if c in ("'", '"'):
                i = ti
                ti += 1
                while text[ti] != c:
                    if text[ti] == "\\":
                        ti += 1
                    ti += 1
                ti += 1
                tok = text[i:ti]
                return

            # number
            if c.isdigit() or (c == "-" and text[ti + 1].isdigit()):
                # integer part
                i = ti
                ti += 1
                while text[ti].isalnum():
                    ti += 1

                # rational
                if text[ti] == "/":
                    ti += 1
                    while text[ti].isdigit():
                        ti += 1

                # real
                else:
                    if text[ti] == ".":
                        ti += 1
                        while text[ti].isalnum():
                            ti += 1
                    if text[ti - 1] in ("e", "E") and text[ti] in ("+", "-"):
                        ti += 1
                        while text[ti].isdigit():
                            ti += 1

                tok = text[i:ti]
                return

            # punctuation
            if text[ti : ti + 3] in ("<=>", "<~>"):
                tok = text[ti : ti + 3]
                ti += 3
                return
            if text[ti : ti + 2] in ("!=", "=>", "<=", "~&", "~|"):
                tok = text[ti : ti + 2]
                ti += 2
                return
            tok = c
            ti += 1
            return

        # end of file
        tok = None

    def eat(o):
        if tok == o:
            lex()
            return True

    def expect(o):
        if tok != o:
            err(f"expected '{o}'")
        lex()

    # terms
    bound = None
    free = {}

    def read_name():
        o = tok

        # word
        if o[0].islower():
            lex()
            return o

        # single quoted, equivalent to word
        if o[0] == "'":
            lex()
            return unquote(o)

        # number
        if o[0].isdigit() or o[0] == "-":
            lex()
            return int(o)

        err("expected name")

    def args(n=-1):
        expect("(")
        r = []
        if tok != ")":
            r.append(atomic_term())
            while tok == ",":
                lex()
                r.append(atomic_term())
        if n > 0 and len(r) != n:
            err(f"expected {n} args")
        expect(")")
        return tuple(r)

    def atomic_term():
        o = tok

        # defined function
        if o[0] == "$":
            if eat("$false"):
                return False
            if eat("$true"):
                return True
            if o in defined_fns:
                a = defined_fns[o]
                lex()
                return (a,) + args(arity(a))
            if eat("$distinct"):
                r = args()
                inequalities = ["and"]
                for i in range(len(r)):
                    for j in range(len(r)):
                        if i != j:
                            inequalities.append(("not", ("=", r[i], r[j])))
                return tuple(inequalities)
            err("unknown function")

        # distinct object
        if o[0] == '"':
            lex()
            return distinct_object(o)

        # number
        if o[0].isdigit() or o[0] == "-":
            lex()
            try:
                return int(o)
            except ValueError:
                if "/" in o:
                    return fractions.Fraction(o)
                return Real(o)

        # variable
        if o[0].isupper():
            lex()
            b = bound
            while b:
                m = b[0]
                if o in m:
                    return m[o]
                b = b[1]
            if o in free:
                return free[o]
            a = Var()
            free[o] = a
            return a

        # function
        name = read_name()
        if tok == "(":
            s = args()
            a = fn(name, Var(None), s)
            return (a,) + s
        return fn(name, Var(None), [])

    def infix_unary():
        a = atomic_term()
        o = tok
        if o == "=":
            lex()
            return "=", a, atomic_term()
        if o == "!=":
            lex()
            return "not", ("=", a, atomic_term())
        return a

    def atomic_type():
        o = tok

        # word
        if o[0].islower():
            lex()
            if o in type_keywords:
                return "'" + o + "'"
            return o

        # defined type
        if o in defined_types:
            lex()
            return defined_types[o]

        # single quoted, equivalent to word
        if o[0] == "'":
            lex()
            u = unquote(o)
            if not weird_type(u):
                return u
            return o

        err("expected type")

    def compound_type():
        if eat("("):
            params = [atomic_type()]
            while eat("*"):
                params.append(atomic_type())
            expect(")")
            expect(">")
            return (atomic_type(),) + tuple(params)
        ty = atomic_type()
        if eat(">"):
            return atomic_type(), ty
        return ty

    def var():
        o = tok
        if not o[0].isupper():
            err("expected variable")
        lex()
        ty = "individual"
        if eat(":"):
            ty = atomic_type()
        a = Var(ty)
        bound[0][o] = a
        return a

    def unitary_formula():
        nonlocal bound
        o = tok
        if o == "(":
            lex()
            a = logic_formula()
            expect(")")
            return a
        if o == "~":
            lex()
            return "not", unitary_formula()
        if o in ("!", "?"):
            o = "exists" if o == "?" else "forall"
            lex()

            # save variable environment
            old = bound
            bound = {}, bound

            # variables
            if tok != "[":
                err("expected '['")
            lex()
            v = []
            v.append(var())
            while tok == ",":
                lex()
                v.append(var())
            if tok != "]":
                err("expected ']'")
            lex()

            # body
            if tok != ":":
                err("expected ':'")
            lex()
            a = o, tuple(v), unitary_formula()

            # restore variable environment
            bound = old

            return a
        return infix_unary()

    def logic_formula():
        a = unitary_formula()
        o = tok
        if o == "&":
            r = ["and", a]
            while eat("&"):
                r.append(unitary_formula())
            return tuple(r)
        if o == "|":
            r = ["or", a]
            while eat("|"):
                r.append(unitary_formula())
            return tuple(r)
        if o == "=>":
            lex()
            return imp(a, unitary_formula())
        if o == "<=":
            lex()
            return imp(unitary_formula(), a)
        if o == "<=>":
            lex()
            return "eqv", a, unitary_formula()
        if o == "<~>":
            lex()
            return "not", ("eqv", a, unitary_formula())
        if o == "~&":
            lex()
            return "not", ("and", a, unitary_formula())
        if o == "~|":
            lex()
            return "not", ("or", a, unitary_formula())
        return a

    # top level
    def ignore():
        if eat("("):
            while not eat(")"):
                ignore()
            return
        lex()

    def selecting(name):
        return select is True or name in select

    def annotated_clause():
        lex()
        expect("(")

        # name
        name = read_name()
        expect(",")

        # role
        role = read_name()
        expect(",")

        # formula
        parens = eat("(")
        neg = []
        pos = []
        while True:
            a = unitary_formula()
            if isinstance(a, tuple) and a[0] == "not":
                neg.append(a[1])
            else:
                pos.append(a)
            if tok != "|":
                break
            lex()
        if selecting(name):
            c = Clause(None, neg, pos)
            c.filename = filename
            c.name = name
            c.role = role
            problem.clauses.append(c)
        if parens:
            expect(")")

        # annotations
        if tok == ",":
            while tok != ")":
                ignore()

        # end
        expect(")")
        expect(".")

    def annotated_formula():
        lex()
        expect("(")

        # name
        name = read_name()
        expect(",")

        # role
        role = read_name()
        expect(",")

        if role == "type":
            parens = 0
            while eat("("):
                parens += 1

            name = read_name()
            expect(":")
            if eat("$tType"):
                # type exists
                pass
            else:
                # function has type
                ty = compound_type()
                if isinstance(ty, tuple):
                    ty = ty[0]
                typecheck(name, ty)
            if tok == ">":
                raise Inappropriate()

            while parens:
                expect(")")
                parens -= 1
        else:
            # formula
            a = logic_formula()
            assert not free
            if selecting(name):
                c = Formula(name, unquantify(a))
                c.filename = filename
                c.role = role
                if role == "conjecture":
                    if hasattr(problem, "conjecture"):
                        raise ValueError("multiple conjectures")
                    problem.conjecture = c
                    a = "not", a
                    c = Formula(name, unquantify(a), c)
                    c.filename = filename
                    c.role = "negated_conjecture"
                problem.formulas.append(c)

        # annotations
        if tok == ",":
            while tok != ")":
                ignore()

        # end
        expect(")")
        expect(".")

    def include():
        lex()
        expect("(")

        # tptp
        tptp = os.getenv("TPTP")
        if not tptp:
            err("TPTP environment variable not set")

        # file
        filename1 = read_name()

        # select
        select1 = select
        if eat(","):
            expect("[")
            select1 = []
            while True:
                name = read_name()
                if selecting(name):
                    select1.append(name)
                if not eat(","):
                    break
            expect("]")

        # include
        read_tptp1(tptp + "/" + filename1, select1)

        # end
        expect(")")
        expect(".")

    lex()
    while tok:
        if tok == "cnf":
            annotated_clause()
            continue
        if tok in ("fof", "tff"):
            annotated_formula()
            continue
        if tok == "include":
            include()
            continue
        err("unknown language")


def read_tptp(filename):
    global problem
    reset_formula_names()
    problem = Problem(filename)
    sys.setrecursionlimit(2000)
    read_tptp1(filename)
    return problem


# print
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


def pr(a):
    print(a, end="")


def prargs(a):
    pr("(")
    for i in range(1, len(a)):
        if i > 1:
            pr(",")
        prterm(a[i])
    pr(")")


def need_parens(a, parent):
    if not parent:
        return
    if a[0] in ("and", "eqv", "or"):
        return parent[0] in ("and", "eqv", "exists", "forall", "not", "or",)


def prterm(a, parent=None):
    if isinstance(a, tuple):
        o = a[0]
        if o == "=":
            prterm(a[1])
            pr("=")
            prterm(a[2])
            return
        if o == "not":
            if isinstance(a[1], tuple) and a[1][0] == "=":
                a = a[1]
                prterm(a[1])
                pr("!=")
                prterm(a[2])
                return
            pr("~")
            prterm(a[1], a)
            return
        if o in ("exists", "forall"):
            if o == "exists":
                pr("?")
            else:
                pr("!")
            pr("[")
            v = a[1]
            for i in range(len(v)):
                if i:
                    pr(",")
                x = v[i]
                set_var_name(x)
                pr(x)
                if x.ty != "individual":
                    pr(":")
                    prtype(x.ty)
            pr("]:")
            prterm(a[2], a)
            return
        connectives = {"and": "&", "eqv": "<=>", "or": "|"}
        if o in connectives:
            if need_parens(a, parent):
                pr("(")
            assert len(a) >= 3
            prterm(a[1], a)
            for i in range(2, len(a)):
                pr(f" {connectives[o]} ")
                prterm(a[i], a)
            if need_parens(a, parent):
                pr(")")
            return
        pr(o)
        prargs(a)
        return
    if a is False:
        pr("$false")
        return
    if a is True:
        pr("$true")
        return
    if isinstance(a, Var):
        set_var_name(a)
    pr(a)


def prtype(a):
    if isinstance(a, str):
        defined_types = {
            "bool": "$o",
            "individual": "$i",
            "int": "$int",
            "rat": "$rat",
            "real": "$real",
        }
        pr(defined_types[a])
        return
    pr(a)


def prformula(c):
    reset_var_names()
    if isinstance(c, Clause):
        pr("cnf")
    else:
        pr("fof")
    pr("(")
    pr(c.name)
    pr(", ")
    if hasattr(c, "role"):
        pr(c.role)
    else:
        pr("plain")
    pr(", ")
    if isinstance(c, Clause):
        i = 0
        for a in c.neg:
            if i:
                pr(" | ")
            i = 1
            prterm(("not", a))
        for a in c.pos:
            if i:
                pr(" | ")
            i = 1
            prterm(a)
        if not c.neg and not c.pos:
            pr("$false")
    else:
        prterm(quantify(c.term))
    print(").")


def prproof(c):
    walk_proof(c, prformula)


######################################## CNF


def cnf(formulas, clauses):
    def skolem(rty, args):
        a = Fn(None, rty, args)
        if args:
            return (a,) + tuple(args)
        return a

    def nnf(all_vars, exists_vars, pol, a):
        if isinstance(a, tuple):
            o = a[0]
            if o == "not":
                return nnf(all_vars, exists_vars, not pol, a[1])

            if o == "and":
                if not pol:
                    o = "or"
                r = [o]
                for b in a[1:]:
                    r.append(nnf(all_vars, exists_vars, pol, b))
                return tuple(r)
            if o == "or":
                if not pol:
                    o = "and"
                r = [o]
                for b in a[1:]:
                    r.append(nnf(all_vars, exists_vars, pol, b))
                return tuple(r)

            if o == "exists":
                if not pol:
                    o = "forall"
                exists_vars = exists_vars.copy()
                for x in a[1]:
                    exists_vars[x] = skolem(x.ty, all_vars.values())
                return nnf(all_vars, exists_vars, pol, a[2])
            if o == "forall":
                if not pol:
                    o = "exists"
                all_vars = all_vars.copy()
                for x in a[1]:
                    all_vars[x] = Var(x.ty)
                return nnf(all_vars, exists_vars, pol, a[2])

            if o == "eqv":
                # a1 => a2
                x = (
                    "or",
                    nnf(all_vars, exists_vars, False, a[1]),
                    nnf(all_vars, exists_vars, pol, a[2]),
                )

                # a1 <= a2
                y = (
                    "or",
                    nnf(all_vars, exists_vars, True, a[1]),
                    nnf(all_vars, exists_vars, not pol, a[2]),
                )

                # and
                return "and", x, y

            r = [a[0]]
            for b in a[1:]:
                r.append(nnf(all_vars, exists_vars, True, b))
            a = tuple(r)
        else:
            if a is False:
                return not pol
            if a is True:
                return pol
            if isinstance(a, Var):
                if a in all_vars:
                    return all_vars[a]
                if a in exists_vars:
                    return exists_vars[a]
                raise ValueError(a)
        return a if pol else ("not", a)

    def rename(a):
        b = skolem("bool", free_vars(a))
        f = Formula(None, imp(b, a))
        convert(f)
        return b

    def distribute(a):
        if isinstance(a, tuple):
            o = a[0]
            if o == "and":
                # flat layer of AND
                r = [o]
                for b in a[1:]:
                    b = distribute(b)
                    if isinstance(b, tuple) and b[0] == "and":
                        r.extend(b[1:])
                        continue
                    r.append(b)
                assert len(r) >= 3
                return tuple(r)
            if o == "or":
                # flat layer of ANDs
                ands = []
                total = 1
                for b in a[1:]:
                    b = distribute(b)
                    if isinstance(b, tuple) and b[0] == "and":
                        n = len(b) - 1
                        if total > 1 and n > 1 and total * n > 4:
                            ands.append([rename(b)])
                            continue
                        ands.append(b[1:])
                        total *= n
                        continue
                    ands.append([b])

                # cartesian product of ANDs
                r = ["and"]
                for c in itertools.product(*ands):
                    r.append((("or",) + tuple(c)))
                if len(r) < 3:
                    return r[1]
                return tuple(r)
        return a

    def split(a, neg, pos):
        if isinstance(a, tuple):
            o = a[0]
            if o == "not":
                neg.append(a[1])
                return
            if o == "or":
                for b in a[1:]:
                    split(b, neg, pos)
                return
            assert o != "and"
        pos.append(a)

    def clause(f, a):
        neg = []
        pos = []
        split(a, neg, pos)
        c = Clause(None, neg, pos, f)
        clauses.append(c)

    def convert(f):
        # variables must be bound only for the first step
        a = quantify(f.term)

        # negation normal form includes several transformations that need to be done together
        b = nnf({}, {}, True, a)
        a = unquantify(a)
        if not isomorphic(a, b, {}):
            f = Formula(None, b, f)
            a = b

        # distribute OR down into AND
        b = distribute(a)
        if a != b:
            f = Formula(None, b, f)
            a = b

        # split AND into clauses
        if isinstance(a, tuple) and a[0] == "and":
            for b in a[1:]:
                clause(f, b)
            return
        clause(f, a)

    for f in formulas:
        convert(f)


######################################## read and prepare


def read_problem(filename):
    # read
    fns.clear()
    problem = read_tptp(filename)

    # infer types
    terms = []
    for f in problem.formulas:
        terms.append(f.term)
    for c in problem.clauses:
        terms.extend(c.neg + c.pos)
    m = {}
    for a in terms:
        type_unify("bool", a, m)
    for a in terms:
        type_set(a, m)
    for a in terms:
        type_check("bool", a)

    # convert to clause normal form
    cnf(problem.formulas, problem.clauses)
    return problem


######################################## test


def test(filename):
    if os.path.splitext(filename)[1] != ".p":
        return
    if "^" in filename:
        return
    print(f"{filename:40s} ", end="", flush=True)
    try:
        start = time.time()
        problem = read_problem(filename)
        print(
            f"{len(problem.formulas):7d} {len(problem.clauses):7d} {time.time()-start:10.3f}"
        )
    except Inappropriate:
        print("Inappropriate")
    except RecursionError:
        print("RecursionError")


def do_file(filename, f):
    if os.path.splitext(filename)[1] == ".lst":
        for s in read_lines(filename):
            do_file(s, f)
        return
    f(filename)


def do_files(files, f):
    for fname in files:
        if os.path.isfile(fname):
            do_file(fname, f)
            continue
        for root, dirs, files in os.walk(fname):
            for filename in files:
                do_file(os.path.join(root, filename), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPTP parser")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    do_files(args.files, test)
