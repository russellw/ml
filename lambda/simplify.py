from etc import *


class Unknown(Exception):
    pass


def len1(a):
    match a:
        case ():
            return 0
        case "quote", s:
            return len(s)
        case "cons", x, s:
            return 1 + len1(s)
        case "cdr", s:
            return max(len1(s) - 1, 0)
    raise Unknown()


def quote(a):
    match a:
        case str():
            return "quote", a
        case ():
            return a
        case *_,:
            return "quote", a
    return a


def unquote(a):
    match a:
        case "quote", x:
            return x
    return a


def simplify(a):
    # an atom is already in simplest form
    if not isinstance(a, tuple):
        return a

    # special form whose arguments cannot be recursively simplified
    match a:
        case "quote", x:
            return quote(x)

    # recur on arguments
    a = tuple(map(simplify, a))

    # mathematical shortcuts
    match a:
        case "or", x, False:
            return x
        case "or", False, x:
            return x
        case "or", _, True:
            return True
        case "or", True, _:
            return True
        case "and", x, True:
            return x
        case "and", True, x:
            return x
        case "and", _, False:
            return False
        case "and", False, _:
            return False
        case "+", x, 0:
            return x
        case "+", 0, x:
            return x
        case "-", x, 0:
            return x
        case "-", x, y:
            if x == y:
                return 0
        case "*", x, 1:
            return x
        case "*", 1, x:
            return x
        case "/", x, 1:
            return x
        case "if", True, x, _:
            return x
        case "if", False, _, x:
            return x
        case "if", _, x, y:
            if x == y:
                return x
        case "==", x, y:
            if x == y:
                return True
        case "<=", x, y:
            if x == y:
                return True
        case "<", x, y:
            if x == y:
                return False
        case "map", _, ():
            return ()
        case "map", ("lambda", (x,), y), s:
            if x == y:
                return s
        case "len", s:
            try:
                return len1(s)
            except Unknown:
                pass

    # are all the arguments constant?
    if not all(map(const, a[1:])):
        return a

    # if so, we can evaluate the term immediately
    match a:
        case ("bool" | "int" | "len" | "not"), x:
            x = unquote(x)
            return eval(f"{a[0]} ({x})")
        case (
            "and" | "or" | "==" | "!=" | "<" | "<=" | "+" | "-" | "*" | "/" | "//" | "%"
        ), x, y:
            x = unquote(x)
            y = unquote(y)
            return eval(f"({x}) {a[0]} ({y})")
    return a
