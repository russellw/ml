import argparse
import fractions
import heapq
import inspect
import itertools
import os
import re
import sys
import time

import psutil

process = psutil.Process()


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


def invert(m):
    r = dict(map(reversed, m.items()))
    assert len(r) == len(m)
    return r


def remove(s, i):
    s = list(s)
    del s[i]
    return tuple(s)


######################################## limits


class MemoryOut(Exception):
    def __init__(self):
        super().__init__("MemoryOut")


class Timeout(Exception):
    def __init__(self):
        super().__init__("Timeout")


def set_timeout(t=60):
    global end_time
    end_time = time.time() + t


def check_limits():
    if process.memory_info().rss > 10_000_000_000:
        raise MemoryOut()
    if time.time() > end_time:
        raise Timeout()


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
skolem_name_i = 0


class Fn:
    def __init__(self, name=None):
        global skolem_name_i
        if name is None:
            skolem_name_i += 1
            name = f"sk{skolem_name_i}"
        else:
            m = re.match(r"sK(\d+)", name)
            if m:
                skolem_name_i = max(skolem_name_i, int(m[1]))
        self.name = name

    def type_args(self, rty, args):
        self.ty = rty
        if args:
            self.ty = (rty,) + tuple([typeof(a) for a in args])

    def __repr__(self):
        return self.name


# TODO: do we need to track functions across problems?
__fns = {}


def clear_fns():
    global skolem_name_i
    __fns.clear()
    skolem_name_i = 0


def fn(name):
    if name in __fns:
        return __fns[name]
    a = Fn(name)
    __fns[name] = a
    return a


# named types are handled as functions
types = {}


def mktype(name):
    if name in types:
        return types[name]
    a = Fn(name)
    types[name] = a
    return a


# first-order variables cannot be boolean
class Var:
    def __init__(self, ty=None):
        if not ty:
            return
        assert ty != "bool"
        self.ty = ty

    def __repr__(self):
        if not hasattr(self, "name"):
            return "Var"
        return self.name


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


def equation_atom(a, b):
    if b is True:
        return a
    return "=", a, b


def fns(a):
    s = set()

    def get_fn(a):
        if isinstance(a, Fn):
            s.add(a)

    walk(get_fn, a)
    return s


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


def match(a, b, m):
    if typeof(a) != typeof(b):
        return
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return
        if a[0] != b[0]:
            return
        for i in range(1, len(a)):
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


def rename_vars(a, m):
    if isinstance(a, tuple):
        return tuple([rename_vars(b, m) for b in a])
    if isinstance(a, Var):
        if a in m:
            return m[a]
        b = Var(a.ty)
        m[a] = b
        return b
    return a


def simplify(a):
    if isinstance(a, tuple):
        a = tuple(map(simplify, a))
        o = a[0]
        if o in ("+", "-", "/", "*"):
            x = a[1]
            y = a[2]
            if const(x) and const(y):
                ty = typeof(x)
                a = eval(f"x{o}y")
                if ty == "real":
                    return Real(a)
            return a
        if o in ("<", "<="):
            x = a[1]
            y = a[2]
            if const(x) and const(y):
                ty = typeof(x)
                return eval(f"x{o}y")
            return a
        if o == "=":
            x = a[1]
            y = a[2]
            if x == y:
                return True
            if unequal(x, y):
                return False
            return a
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


def walk(f, a):
    if isinstance(a, tuple):
        for b in a:
            walk(f, b)
    f(a)


######################################## types


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
    if isinstance(a, Fn) or isinstance(a, Var):
        return a.ty
    if isinstance(a, bool):
        return "bool"
    if isinstance(a, DistinctObject):
        return "individual"
    if isinstance(a, int):
        return "int"
    # as subclass, Real must be checked before Fraction
    if isinstance(a, Real):
        return "real"
    if isinstance(a, fractions.Fraction):
        return "rat"
    raise ValueError(a)


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
                    type_check("int", a[i])

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


######################################## logic


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


class Formula:
    def __init__(self, name, term, inference=None, *parents):
        set_formula_name(self, name)
        self.__term = term
        self.inference = inference
        self.parents = parents

    def proof(self):
        visited = set()
        r = []

        def rec(c):
            if c in visited:
                return
            visited.add(c)
            for d in c.parents:
                rec(d)
            r.append(c)

        rec(self)
        return r

    def status(self):
        # input data or definition is logical data
        if not self.parents:
            return "lda"

        # negated conjecture is counterequivalent
        if self.inference == "negate":
            return "ceq"

        # if a formula introduces new symbols, then it is only equisatisfiable
        # this happens during subformula renaming in CNF conversion
        if len(self.parents) == 1 and not fns(self.term()).issubset(
            fns(self.parents[0].term())
        ):
            return "esa"

        # formula is a theorem of parents
        # could also be equivalent; don't currently bother distinguishing that case
        return "thm"

    def term(self):
        return self.__term


class Clause(Formula):
    def __init__(self, name, neg, pos, inference=None, *parents):
        for a in neg:
            check_tuples(a)
        for a in pos:
            check_tuples(a)
        set_formula_name(self, name)
        self.neg = tuple(neg)
        self.pos = tuple(pos)
        self.inference = inference
        self.parents = parents

    def __lt__(self, other):
        return self.size() < other.size()

    def __repr__(self):
        return str(self.neg) + "=>" + str(self.pos)

    def rename_vars(self):
        m = {}
        neg = [rename_vars(a, m) for a in self.neg]
        pos = [rename_vars(a, m) for a in self.pos]
        c = Clause("*RENAMED*", neg, pos, "rename_vars", self)
        return c

    def simplify(self):
        # simplify terms
        neg = map(simplify, self.neg)
        pos = map(simplify, self.pos)

        # eliminate redundancy
        neg = filter(lambda a: a != True, neg)
        pos = filter(lambda a: a != False, pos)

        # reify iterators
        neg = tuple(neg)
        pos = tuple(pos)

        # check for tautology
        if False in neg or True in pos:
            neg, pos = (), (True,)
        else:
            for a in neg:
                if a in pos:
                    neg, pos = (), (True,)

        # did anything change?
        if (neg, pos) == (self.neg, self.pos):
            return self

        # derived clause
        return Clause(None, neg, pos, "simplify", self)

    def size(self):
        return term_size(self.neg + self.pos)

    def term(self):
        r = tuple([("not", a) for a in self.neg]) + self.pos
        if not r:
            return False
        if len(r) == 1:
            return r[0]
        return ("or",) + r


class Problem:
    def __init__(self, name):
        self.name = name
        self.formulas = []
        self.clauses = []
        self.expected = None


######################################## TPTP


defined_types = {
    "$o": "bool",
    "$i": "individual",
    "$int": "int",
    "$rat": "rat",
    "$real": "real",
}

defined_fns = {
    "$ceiling": "ceil",
    "$difference": "-",
    "$floor": "floor",
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


# parser
class Inappropriate(Exception):
    def __init__(self):
        super().__init__("Inappropriate")


def read_tptp1(filename, select=True):
    global expected
    global header
    fname = os.path.basename(filename)
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
            if c in ("%", "#"):
                i = ti
                while text[ti] != "\n":
                    ti += 1
                if header:
                    print(text[i:ti])
                    if text[ti : ti + 2] == "\n\n":
                        print()
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
    def read_name():
        o = tok

        # word
        if o[0].islower():
            lex()
            return o

        # single quoted, equivalent to word
        if o[0] == "'":
            lex()
            return o[1:-1]

        # number
        if o[0].isdigit() or o[0] == "-":
            lex()
            return int(o)

        err("expected name")

    def atomic_type():
        o = tok
        if o in defined_types:
            lex()
            return defined_types[o]
        if tok == "$tType":
            raise Inappropriate()
        return mktype(read_name())

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

    free = {}

    def args(bound, n=-1):
        expect("(")
        r = []
        if tok != ")":
            r.append(atomic_term(bound))
            while tok == ",":
                lex()
                r.append(atomic_term(bound))
        if n > 0 and len(r) != n:
            err(f"expected {n} args")
        expect(")")
        return tuple(r)

    def atomic_term(bound):
        o = tok

        # defined function
        if o[0] == "$":
            # constant
            if eat("$false"):
                return False
            if eat("$true"):
                return True

            # syntax sugar
            if eat("$greater"):
                s = args(bound, 2)
                return "<", s[1], s[0]
            if eat("$greatereq"):
                s = args(bound, 2)
                return "<=", s[1], s[0]
            if eat("$distinct"):
                s = args()
                inequalities = ["and"]
                for i in range(len(s)):
                    for j in range(len(s)):
                        if i != j:
                            inequalities.append(("not", ("=", s[i], s[j])))
                return tuple(inequalities)

            # predefined function
            if o in defined_fns:
                a = defined_fns[o]
                lex()
                arities = {
                    "*": 2,
                    "+": 2,
                    "-": 2,
                    "/": 2,
                    "<": 2,
                    "<=": 2,
                    "ceil": 1,
                    "div-e": 2,
                    "div-f": 2,
                    "div-t": 2,
                    "floor": 1,
                    "int?": 1,
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
                return (a,) + args(bound, arities[a])
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
            if o in bound:
                return bound[o]
            if o in free:
                return free[o]
            a = Var("individual")
            free[o] = a
            return a

        # higher-order terms
        if tok == "!":
            raise Inappropriate()

        # function
        a = fn(read_name())
        if tok == "(":
            s = args(bound)
            if not hasattr(a, "ty"):
                a.type_args(Var(), s)
            return (a,) + s
        if not hasattr(a, "ty"):
            a.ty = Var()
        return a

    def infix_unary(bound):
        a = atomic_term(bound)
        o = tok
        if o == "=":
            lex()
            return "=", a, atomic_term(bound)
        if o == "!=":
            lex()
            return "not", ("=", a, atomic_term(bound))
        return a

    def var(bound):
        o = tok
        if not o[0].isupper():
            err("expected variable")
        lex()
        ty = "individual"
        if eat(":"):
            ty = atomic_type()
        a = Var(ty)
        bound[o] = a
        return a

    def unitary_formula(bound):
        o = tok
        if o == "(":
            lex()
            a = logic_formula(bound)
            expect(")")
            return a
        if o == "~":
            lex()
            return "not", unitary_formula(bound)
        if o in ("!", "?"):
            o = "exists" if o == "?" else "forall"
            lex()

            # variables
            bound = bound.copy()
            expect("[")
            v = []
            v.append(var(bound))
            while tok == ",":
                lex()
                v.append(var(bound))
            expect("]")

            # body
            expect(":")
            a = o, tuple(v), unitary_formula(bound)
            return a
        return infix_unary(bound)

    def logic_formula(bound):
        a = unitary_formula(bound)
        o = tok
        if o == "&":
            r = ["and", a]
            while eat("&"):
                r.append(unitary_formula(bound))
            return tuple(r)
        if o == "|":
            r = ["or", a]
            while eat("|"):
                r.append(unitary_formula(bound))
            return tuple(r)
        if o == "=>":
            lex()
            return imp(a, unitary_formula(bound))
        if o == "<=":
            lex()
            return imp(unitary_formula(bound), a)
        if o == "<=>":
            lex()
            return "eqv", a, unitary_formula(bound)
        if o == "<~>":
            lex()
            return "not", ("eqv", a, unitary_formula(bound))
        if o == "~&":
            lex()
            return "not", ("and", a, unitary_formula(bound))
        if o == "~|":
            lex()
            return "not", ("or", a, unitary_formula(bound))
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
            a = unitary_formula({})
            if isinstance(a, tuple) and a[0] == "not":
                neg.append(a[1])
            else:
                pos.append(a)
            if tok != "|":
                break
            lex()
        if selecting(name):
            c = Clause(name, neg, pos)
            c.fname = fname
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
            # type declaration
            parens = 0
            while eat("("):
                parens += 1

            name = read_name()
            expect(":")
            if eat("$tType"):
                # type exists
                if tok == ">":
                    raise Inappropriate()
            else:
                if tok in ("!", "["):
                    raise Inappropriate()
                # function has type
                a = fn(name)
                ty = compound_type()
                if not hasattr(a, "ty"):
                    a.ty = ty
                else:
                    if a.ty != ty:
                        err("type mismatch")

            while parens:
                expect(")")
                parens -= 1
        else:
            # formula
            a = logic_formula({})
            assert not free
            if selecting(name):
                F = Formula(name, unquantify(a))
                F.fname = fname
                F.role = role
                if role == "conjecture":
                    if hasattr(problem, "conjecture"):
                        err("multiple conjectures")
                    problem.conjecture = F
                    F = Formula(name, ("not", a), "negate", F)
                    F.role = "negated_conjecture"
                problem.formulas.append(F)

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
    header = False
    while tok:
        free.clear()
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
    global header
    global problem
    header = True
    problem = Problem(filename)
    reset_formula_names()
    # numbers larger than 2000 silently fail
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
        # infix
        if o == "=":
            prterm(a[1])
            pr("=")
            prterm(a[2])
            return
        connectives = {"and": "&", "eqv": "<=>", "or": "|"}
        if o in connectives:
            if need_parens(a, parent):
                pr("(")
            assert len(a) >= 3
            for i in range(1, len(a)):
                if i > 1:
                    pr(f" {connectives[o]} ")
                prterm(a[i], a)
            if need_parens(a, parent):
                pr(")")
            return

        # prefix/infix
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

        # prefix
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
        if isinstance(o, str):
            pr(invert(defined_fns)[o])
        else:
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
        pr(invert(defined_types)[a])
        return
    pr(a)


def prformula(c):
    reset_var_names()
    if isinstance(c, Clause):
        pr("cnf")
    else:
        pr("fof")
    pr("(")

    # name
    pr(c.name)
    pr(", ")

    # role
    if hasattr(c, "role"):
        pr(c.role)
    else:
        pr("plain")
    pr(", ")

    # content
    a = c.term()
    if isinstance(c, Formula):
        a = quantify(a)
    prterm(a)
    pr(", ")

    # source
    if hasattr(c, "fname"):
        pr(f"file('{c.fname}',{c.name})")
    elif c.inference:
        pr(f"inference({c.inference},[status({c.status()})],[")
        for i in range(len(c.parents)):
            if i:
                pr(",")
            pr(c.parents[i].name)
        pr("])")
    else:
        pr("introduced(definition)")

    # end
    print(").")


######################################## CNF


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


def cnf(formulas, clauses):
    def skolem(rty, args):
        a = Fn()
        a.type_args(rty, args)
        if args:
            return (a,) + tuple(args)
        return a

    def nnf(all_vars, exists_vars, pol, a):
        check_limits()
        if isinstance(a, tuple):
            o = a[0]
            if o == "not":
                return nnf(all_vars, exists_vars, not pol, a[1])
            if o == "and":
                if not pol:
                    o = "or"
                return (o,) + tuple([nnf(all_vars, exists_vars, pol, b) for b in a[1:]])
            if o == "or":
                if not pol:
                    o = "and"
                return (o,) + tuple([nnf(all_vars, exists_vars, pol, b) for b in a[1:]])
            if o in ("exists", "forall"):
                if not pol:
                    if o == "exists":
                        o = "forall"
                    else:
                        o = "exists"
                if o == "exists":
                    exists_vars = exists_vars.copy()
                    for x in a[1]:
                        exists_vars[x] = skolem(x.ty, all_vars.values())
                else:
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
        convert(Formula(None, imp(b, a)))
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

    def clause(F, a):
        neg = []
        pos = []
        split(a, neg, pos)
        clauses.append(Clause(None, neg, pos, "split", F))

    def convert(F):
        # variables must be bound only for the first step
        a = quantify(F.term())

        # negation normal form includes several transformations that need to be done together
        b = nnf({}, {}, True, a)
        a = unquantify(a)
        if not isomorphic(a, b, {}):
            F = Formula(None, b, "nnf", F)
            a = b

        # distribute OR down into AND
        b = distribute(a)
        if a != b:
            F = Formula(None, b, "distribute", F)
            a = b

        # split AND into clauses
        if isinstance(a, tuple) and a[0] == "and":
            for b in a[1:]:
                clause(F, b)
            return
        clause(F, a)

    for F in formulas:
        convert(F)


######################################## read and prepare


def read_problem(filename):
    # read
    clear_fns()
    problem = read_tptp(filename)

    # infer types
    terms = [c.term() for c in problem.formulas + problem.clauses]
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


######################################## subsumption


def subsumes(c, d):
    # negative and positive literals must subsume separately
    c1 = c.neg
    c2 = c.pos
    d1 = d.neg
    d2 = d.pos

    # longer clause cannot subsume shorter one
    if len(c1) > len(d1) or len(c2) > len(d2):
        return

    # fewer literals typically fail faster, so try fewer side first
    if len(c2) + len(d2) < len(c1) + len(d1):
        c1, c2 = c2, c1
        d1, d2 = d2, d1

    # search with timeout
    steps = 0

    def search(c1, c2, d1, d2, m):
        nonlocal steps

        # worst-case time is exponential
        # so give up if taking too long
        if steps == 1000:
            raise Timeout()
        steps += 1

        # matched everything in one polarity?
        if not c1:
            # matched everything in the other polarity?
            if not c2:
                return m

            # try the other polarity
            return search(c2, None, d2, None, m)

        # try matching literals
        for ci in range(len(c1)):
            ce = equation(c1[ci])
            for di in range(len(d1)):
                de = equation(d1[di])

                # try orienting equation one way
                m1 = m.copy()
                if match(ce[1], de[1], m1) and match(ce[2], de[2], m1):
                    m1 = search(remove(c1, ci), c2, remove(d1, di), d2, m1)
                    if m1 is not None:
                        return m1

                # and the other way
                m1 = m.copy()
                if match(ce[1], de[2], m1) and match(ce[2], de[1], m1):
                    m1 = search(remove(c1, ci), c2, remove(d1, di), d2, m1)
                    if m1 is not None:
                        return m1

    try:
        m = search(c1, c2, d1, d2, {})
        return m is not None
    except Timeout:
        pass


def forward_subsumes(ds, c):
    for d in ds:
        if hasattr(d, "dead"):
            continue
        if subsumes(d, c):
            return True


def backward_subsume(c, ds):
    for d in ds:
        if hasattr(d, "dead"):
            continue
        if subsumes(c, d):
            d.dead = True


######################################## superposition


# partial implementation of the superposition calculus
# a full implementation would also implement an order on equations
# e.g. lexicographic path ordering or Knuth-Bendix ordering
def original(c):
    if c.inference == "rename_vars":
        return c.parents[0]
    return c


def clause(m, neg, pos, inference, *parents):
    check_limits()
    neg = subst(tuple(neg), m)
    pos = subst(tuple(pos), m)
    c = Clause(None, neg, pos, inference, *map(original, parents)).simplify()
    if c.term() is True:
        return
    if c.size() > 10_000_000:
        raise ResourceOut()
    clauses.append(c)


# equality resolution
# c | c0 != c1
# ->
# c/m
# where
# m = unify(c0, c1)

# for each negative equation
def resolution(c):
    for ci in range(len(c.neg)):
        _, c0, c1 = equation(c.neg[ci])
        m = {}
        if unify(c0, c1, m):
            resolutionc(c, ci, m)


# substitute and make new clause
def resolutionc(c, ci, m):
    neg = remove(c.neg, ci)
    pos = c.pos
    clause(m, neg, pos, "resolve", c)


# equality factoring
# c | c0 = c1 | c2 = c3
# ->
# (c | c0 = c1 | c1 != c3)/m
# where
# m = unify(c0, c2)

# for each positive equation (both directions)
def factoring(c):
    for ci in range(len(c.pos)):
        _, c0, c1 = equation(c.pos[ci])
        factoring1(c, ci, c0, c1)
        factoring1(c, ci, c1, c0)


# for each positive equation (both directions) again
def factoring1(c, ci, c0, c1):
    for cj in range(len(c.pos)):
        if cj == ci:
            continue
        _, c2, c3 = equation(c.pos[cj])
        factoringc(c, c0, c1, cj, c2, c3)
        factoringc(c, c0, c1, cj, c3, c2)


# check, substitute and make new clause
def factoringc(c, c0, c1, cj, c2, c3):
    if not equatable(c1, c3):
        return
    m = {}
    if not unify(c0, c2, m):
        return
    neg = c.neg + (equation_atom(c1, c3),)
    pos = remove(c.pos, cj)
    clause(m, neg, pos, "factor", c)


# negative superposition
# c | c0 = c1, d | d0(a) != d1
# ->
# (c | d | d0(c1) != d1)/m
# where
# m = unify(c0, a)
# a not variable

# for each positive equation in c (both directions)
def superposition_neg(c, d):
    for ci in range(len(c.pos)):
        _, c0, c1 = equation(c.pos[ci])
        superposition_neg1(c, d, ci, c0, c1)
        superposition_neg1(c, d, ci, c1, c0)


# for each negative equation in d (both directions)
def superposition_neg1(c, d, ci, c0, c1):
    if c0 is True:
        return
    for di in range(len(d.neg)):
        _, d0, d1 = equation(d.neg[di])
        superposition_neg2(c, d, ci, c0, c1, di, d0, d1, [], d0)
        superposition_neg2(c, d, ci, c0, c1, di, d1, d0, [], d1)


# descend into subterms
def superposition_neg2(c, d, ci, c0, c1, di, d0, d1, path, a):
    if isinstance(a, Var):
        return
    superposition_negc(c, d, ci, c0, c1, di, d0, d1, path, a)
    if isinstance(a, tuple):
        for i in range(1, len(a)):
            path.append(i)
            superposition_negc(c, d, ci, c0, c1, di, d0, d1, path, a[i])
            path.pop()


# check, substitute and make new clause
def superposition_negc(c, d, ci, c0, c1, di, d0, d1, path, a):
    m = {}
    if not unify(c0, a, m):
        return
    neg = c.neg + remove(d.neg, di) + (equation_atom(splice(d0, path, c1), d1),)
    pos = remove(c.pos, ci) + d.pos
    clause(m, neg, pos, "ns", original(c), original(d))


# positive superposition
# c | c0 = c1, d | d0(a) = d1
# ->
# (c | d | d0(c1) = d1)/m
# where
# m = unify(c0, a)
# a not variable

# for each positive equation in c (both directions)
def superposition_pos(c, d):
    for ci in range(len(c.pos)):
        _, c0, c1 = equation(c.pos[ci])
        superposition_pos1(c, d, ci, c0, c1)
        superposition_pos1(c, d, ci, c1, c0)


# for each positive equation in d (both directions)
def superposition_pos1(c, d, ci, c0, c1):
    if c0 is True:
        return
    for di in range(len(d.pos)):
        _, d0, d1 = equation(d.pos[di])
        superposition_pos2(c, d, ci, c0, c1, di, d0, d1, [], d0)
        superposition_pos2(c, d, ci, c0, c1, di, d1, d0, [], d1)


# descend into subterms
def superposition_pos2(c, d, ci, c0, c1, di, d0, d1, path, a):
    if isinstance(a, Var):
        return
    superposition_posc(c, d, ci, c0, c1, di, d0, d1, path, a)
    if isinstance(a, tuple):
        for i in range(1, len(a)):
            path.append(i)
            superposition_posc(c, d, ci, c0, c1, di, d0, d1, path, a[i])
            path.pop()


# check, substitute and make new clause
def superposition_posc(c, d, ci, c0, c1, di, d0, d1, path, a):
    m = {}
    if not unify(c0, a, m):
        return
    neg = c.neg + d.neg
    pos = (
        remove(c.pos, ci)
        + remove(d.pos, di)
        + (equation_atom(splice(d0, path, c1), d1),)
    )
    clause(m, neg, pos, "ps", original(c), original(d))


# superposition is incomplete on arithmetic
def contains_arithmetic(a):
    if isinstance(a, tuple):
        for b in a[1:]:
            if contains_arithmetic(b):
                return True
    return typeof(a) in ("int", "rat", "real")


def solve(cs):
    global clauses
    unprocessed = [c.simplify() for c in cs]
    heapq.heapify(unprocessed)
    processed = []
    while unprocessed:
        # given clause
        g = heapq.heappop(unprocessed)

        # subsumption
        if hasattr(g, "dead"):
            continue

        # solved?
        if g.term() is False:
            return "Unsatisfiable", g

        # match/unify assume clauses have disjoint variable names
        c = g.rename_vars()

        # subsumption
        if forward_subsumes(processed, c):
            continue
        if forward_subsumes(unprocessed, c):
            continue
        backward_subsume(c, processed)
        backward_subsume(c, unprocessed)

        # may need to match g with itself
        processed.append(g)

        # generate new clauses
        clauses = []
        resolution(c)
        factoring(c)
        for d in processed:
            if hasattr(d, "dead"):
                continue
            superposition_neg(c, d)
            superposition_neg(d, c)
            superposition_pos(c, d)
            superposition_pos(d, c)
        for c in clauses:
            heapq.heappush(unprocessed, c)
    for c in cs:
        for a in c.neg + c.pos:
            if contains_arithmetic(a):
                return "GaveUp", None
    return "Satisfiable", None


######################################## top level


def do_file(filename):
    global attempted
    global solved

    # list file
    if os.path.splitext(filename)[1] == ".lst":
        for s in open(filename).readlines():
            do_file(s.strip())
        return

    # try to solve
    start = time.time()
    set_timeout()
    attempted += 1
    fname = os.path.basename(filename)
    try:
        problem = read_problem(filename)
        if problem.formulas:
            print(f"% {len(problem.formulas)} formulas")
        print(f"% {len(problem.clauses)} clauses")
        r, conclusion = solve(problem.clauses)
        if hasattr(problem, "conjecture"):
            if r == "Satisfiable":
                r = "CounterSatisfiable"
            elif r == "Unsatisfiable":
                r = "Theorem"
        print(f"% SZS status {r} for {fname}")
        if conclusion:
            for c in conclusion.proof():
                prformula(c)
        if r in (
            "Theorem",
            "Unsatisfiable",
            "ContradictoryAxioms",
            "Satisfiable",
            "CounterSatisfiable",
        ):
            solved += 1
            if problem.expected and r != problem.expected:
                if problem.expected == "ContradictoryAxioms" and r in (
                    "Theorem",
                    "Unsatisfiable",
                ):
                    pass
                else:
                    raise ValueError(f"{r} != {problem.expected}")
    except (Inappropriate, Timeout) as e:
        print(f"% SZS status {e} for {fname}")
    except RecursionError:
        print(f"% SZS status ResourceOut for {fname}")
    print(f"% {time.time() - start:.3f} seconds")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="theorem prover")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    start = time.time()
    attempted = 0
    solved = 0
    for filename in args.files:
        if os.path.isfile(filename):
            do_file(filename)
            continue
        for root, dirs, files in os.walk(filename):
            for fname in files:
                do_file(os.path.join(root, fname))
    print(f"% solved {solved}/{attempted} = {solved*100/attempted}%")
    print(f"% {time.time() - start:.3f} seconds")
