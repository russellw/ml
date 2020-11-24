import argparse
import fractions
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


########################################logic


class DistinctObject:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name


# real number constants must be rational, but separate type from Fraction
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
            check_tuples(a)
        for a in pos:
            check_tuples(a)

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
            check_tuples(a)
        for a in pos:
            check_tuples(a)

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


########################################CNF


def check_skolemi(a):
    global skolemi
    if isinstance(a, tuple):
        for b in a:
            check_skolemi(b)
        return
    if isinstance(a, str):
        if a[:2] == "sK":
            try:
                skolemi = max(skolemi, int(a[2:]) + 1)
            except ValueError:
                pass
        return


def skolem(ty, v):
    global skolemi
    f = "sK" + str(skolemi)
    skolemi += 1
    typecheck(f, ty)
    if v:
        return (f,) + tuple(v)
    return f


def nnf(all_vars, exists_vars, pol, a):
    if isinstance(a, tuple):
        o = a[0]
        if o == "not":
            return nnf(all_vars, exists_vars, not pol, a[1])
        if o == "=>":
            return nnf(all_vars, exists_vars, pol, ("or", ("not", a[1]), a[2]))
        if o in ("and", "or"):
            if not pol:
                o = "and" if o == "or" else "or"
            r = [o]
            for b in a[1:]:
                r.append(nnf(all_vars, exists_vars, pol, b))
            return tuple(r)
        if o in ("exists", "forall"):
            if not pol:
                o = "exists" if o == "forall" else "forall"
            if o == "exists":
                exists_vars = dict(exists_vars)
                for x in a[1]:
                    exists_vars[x] = skolem(x.ty, all_vars.values())
            else:
                all_vars = all_vars.copy()
                for x in a[1]:
                    all_vars[x] = Var(x.ty)
            return nnf(all_vars, exists_vars, pol, a[2])
        if o in ("equiv", "xor"):
            if o == "xor":
                pol = not pol

            # a1 => a2
            b = (
                "or",
                nnf(all_vars, exists_vars, False, a[1]),
                nnf(all_vars, exists_vars, pol, a[2]),
            )

            # a1 <= a2
            c = (
                "or",
                nnf(all_vars, exists_vars, True, a[1]),
                nnf(all_vars, exists_vars, not pol, a[2]),
            )

            # and
            return "and", b, c
        r = [a[0]]
        for b in a[1:]:
            r.append(nnf(all_vars, exists_vars, True, b))
        a = tuple(r)
    else:
        if isinstance(a, Var):
            if a in all_vars:
                return all_vars[a]
            if a in exists_vars:
                return exists_vars[a]
            raise ValueError(a)
        if a is False:
            return not pol
        if a is True:
            return pol
    return a if pol else ("not", a)


def rename(a):
    b = skolem("bool", free_vars(a))
    f = Formula(None, ("=>", b, a))
    cnf1(f)
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
        assert o != "and"
        if o == "or":
            for b in a[1:]:
                split(b, neg, pos)
            return
        if o == "not":
            neg.append(a[1])
            return
    pos.append(a)


def clause(f, a):
    neg = []
    pos = []
    split(a, neg, pos)
    c = Clause(None, neg, pos, f)
    clauses.append(c)


def cnf1(f):
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


def cnf(problem):
    global clauses
    global skolemi

    # check if existing function names would clash with Skolem function names
    skolemi = 0
    for f in problem.formulas:
        check_skolemi(f.term)
    for c in problem.clauses:
        for a in c.neg:
            check_skolemi(a)
        for a in c.pos:
            check_skolemi(a)

    # convert
    clauses = problem.clauses
    for f in problem.formulas:
        cnf1(f)


########################################TPTP


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


def read1(filename, select=True):
    global expected
    text = open(filename).read()
    if text and text[-1] != "\n":
        text += "\n"

    # tokenizer

    punct2 = {"/*", "!=", "=>", "<=", "~&", "~|"}
    punct3 = {"<=>", "<~>"}
    quotes = {"'", '"'}

    line = 1
    ti = 0
    tok = ""

    def err(msg):
        raise ValueError(f"{filename}:{line}: {repr(tok)}: {msg}")

    def lex():
        global expected
        nonlocal line
        nonlocal ti
        nonlocal tok
        while ti < len(text):
            c = text[ti]

            # space
            if c == "\n":
                line += 1
                ti += 1
                continue
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

            # word
            if c.isalpha() or c == "$":
                i = ti
                ti += 1
                while text[ti].isalnum() or text[ti] == "_":
                    ti += 1
                tok = text[i:ti]
                return

            # quote
            if c in quotes:
                i = ti
                ti += 1
                while text[ti] != c:
                    if text[ti] == "\\":
                        ti += 1
                    ti += 1
                ti += 1
                tok = text[i:ti]
                return

            # punctuation
            if text[ti : ti + 3] in punct3:
                tok = text[ti : ti + 3]
                ti += 3
                return
            if text[ti : ti + 2] in punct2:
                # block comment
                if text[ti : ti + 2] == "/*":
                    ti += 2
                    while text[ti : ti + 2] != "*/":
                        ti += 1
                    ti += 2
                    continue

                # punctuation
                tok = text[ti : ti + 2]
                ti += 2
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
            tok = c
            ti += 1
            return

        # end of file
        tok = ""

    def eat(o):
        if tok == o:
            lex()
            return True

    def expect(o):
        if tok != o:
            err("expected '" + o + "'")
        lex()

    # terms

    bound = None
    free = {}

    def atom():
        o = tok

        # word
        if o[0].islower():
            lex()
            if o in term_keywords:
                return "'" + o + "'"
            return o

        # single quoted, equivalent to word
        if o[0] == "'":
            lex()
            u = unquote(o)
            if not weird(u):
                return u
            return o

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

        # distinct object
        if o[0] == '"':
            lex()
            return DistinctObject(o)

        # number
        if o[0].isdigit() or o[0] == "-":
            lex()
            try:
                return int(o)
            except ValueError:
                if "/" in o:
                    return fractions.Fraction(o)
                return Real(o)

        err("expected term")

    def args():
        if tok != "(":
            err("expected '('")
        lex()
        r = []
        if tok != ")":
            r.append(atomic_term())
            while tok == ",":
                lex()
                r.append(atomic_term())
        if tok != ")":
            err("expected ')'")
        lex()
        return r

    def atomic_term():
        o = tok
        if o[0] == "$":
            if o in defined_fns:
                lex()
                return (defined_fns[o],) + tuple(args())
            if eat("$distinct"):
                r = args()
                inequalities = ["and"]
                for i in range(len(r)):
                    for j in range(len(r)):
                        if i != j:
                            inequalities.append(("not", ("=", r[i], r[j])))
                return tuple(inequalities)
            if eat("$false"):
                return False
            if eat("$true"):
                return True
            err("unknown function")
        a = atom()
        if tok == "(":
            return (a,) + tuple(args())
        return a

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
            r = [atomic_type()]
            while eat("*"):
                r.append(atomic_type())
            expect(")")
            expect(">")
            return (atomic_type(),) + tuple(r)
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
        if tok == ":":
            lex()
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
            if tok != ")":
                err("expected ')'")
            lex()
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
            return o, a, unitary_formula()
        if o == "<=":
            lex()
            return "=>", unitary_formula(), a
        if o == "<=>":
            lex()
            return "equiv", a, unitary_formula()
        if o == "<~>":
            lex()
            return "xor", a, unitary_formula()
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
        if eat("["):
            while not eat("]"):
                ignore()
            return
        lex()

    def selecting(name):
        return select is True or name in select

    def annotated_clause():
        atom()
        if tok != "(":
            err("expected '('")
        lex()

        # name
        name = atom()
        if tok != ",":
            err("expected ','")
        lex()

        # role
        role = atom()
        if tok != ",":
            err("expected ','")
        lex()

        # formula
        p = False
        if tok == "(":
            lex()
            p = True
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
        if p:
            if tok != ")":
                err("expected ')'")
            lex()

        # annotations
        if tok == ",":
            while tok != ")":
                ignore()

        # end
        if tok != ")":
            err("expected ')'")
        lex()
        if tok != ".":
            err("expected '.'")
        lex()

    def annotated_formula():
        atom()
        if tok != "(":
            err("expected '('")
        lex()

        # name
        name = atom()
        if tok != ",":
            err("expected ','")
        lex()

        # role
        role = atom()
        if tok != ",":
            err("expected ','")
        lex()

        if role == "type":
            p = 0
            while eat("("):
                p += 1

            name = atom()
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

            while p:
                expect(")")
                p -= 1
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
        if tok != ")":
            err("expected ')'")
        lex()
        if tok != ".":
            err("expected '.'")
        lex()

    def include():
        atom()
        expect("(")

        # tptp
        tptp = os.getenv("TPTP")
        if not tptp:
            err("TPTP environment variable not set")

        # file
        filename1 = unquote(atom())

        # select
        select1 = select
        if eat(","):
            expect("[")
            select1 = []
            while True:
                name = atom()
                if selecting(name):
                    select1.append(name)
                if not eat(","):
                    break
            expect("]")

        # include
        read1(tptp + "/" + filename1, select1)

        # end
        expect(")")
        expect(".")

    lex()
    fof_languages = ("fof", "tff")
    while tok:
        if tok in fof_languages:
            annotated_formula()
            continue
        if tok == "cnf":
            annotated_clause()
            continue
        if tok == "include":
            include()
            continue
        err("unknown language")


def read(filename):
    global problem
    fn_types.clear()
    reset_formula_names()
    problem = Problem(filename)
    sys.setrecursionlimit(2000)
    read1(filename)
    cnf(problem)
    return problem


# print

defined_fns_inv = {v: k for k, v in defined_fns.items()}
defined_types_inv = {v: k for k, v in defined_types.items()}


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
    if a[0] in ("=>", "and", "equiv", "or", "xor"):
        return parent[0] in (
            "=>",
            "and",
            "equiv",
            "exists",
            "forall",
            "not",
            "or",
            "xor",
        )


def prterm(a, parent=None):
    if isinstance(a, tuple):
        o = a[0]
        if o in defined_fns_inv:
            pr(defined_fns_inv[o])
            prargs(a)
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
        connectives = {"=>": "=>", "and": "&", "equiv": "<=>", "or": "|", "xor": "<~>"}
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
    if a in defined_types_inv:
        pr(defined_types_inv[a])
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


# test


def test(filename):
    if os.path.splitext(filename)[1] not in (".ax", ".p"):
        return
    if "^" in filename:
        return
    print(f"{filename:40s} ", end="", flush=True)
    try:
        start = time.time()
        problem = read(filename)
        print(
            f"{len(problem.formulas):7d} {len(problem.clauses):7d} {time.time()-start:10.3f}"
        )
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
