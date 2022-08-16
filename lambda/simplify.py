from etc import *


def quote(a):
    if not const(a):
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
            if const(x):
                return x
            return a

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
        case "*", _, 0:
            return 0
        case "*", 0, _:
            return 0
        case "*", x, 1:
            return x
        case "*", 1, x:
            return x
        case "/", x, 1:
            return x
        case "div", x, 1:
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
        case "len", ("cons", _, ()):
            return 1
        case "len", ("cons", _, ("cons", _, ())):
            return 2
        case "len", ("cons", _, ("cons", _, ("cons", _, ()))):
            return 3

    # are all the arguments constant?
    if not all(map(const, a[1:])):
        return a

    # if so, we can evaluate the term immediately
    match a:
        case "not", x:
            return eval(f"{a[0]} {x}")
        case (
            "and" | "or" | "==" | "<" | "<=" | "+" | "-" | "*" | "/" | "//" | "%" | "**"
        ), x, y:
            x = unquote(x)
            y = unquote(y)
            return eval(f"{x} {a[0]} {y}")
        case "cons", x, s:
            x = unquote(x)
            s = unquote(s)
            return quote((x,) + s)
        case "car", s:
            s = unquote(s)
            return quote(s[0])
        case "cdr", s:
            s = unquote(s)
            return quote(s[1:])
        case "len", s:
            s = unquote(s)
            return len(s)
    return a
