from etc import *
import interpreter


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
    if not isinstance(a, tuple) or not a:
        return a

    # special form whose arguments cannot be recursively simplified
    match a:
        case "quote", x:
            return quote(x)
        case "lambda", params, body:
            return "lambda", params, simplify(body)

    # recur on arguments
    a = tuple(map(simplify, a))

    # mathematical shortcuts
    match a:
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
        case "if", 1, x, _:
            return x
        case "if", 0, _, x:
            return x
        case "if", _, x, y:
            if x == y:
                return x
        case "==", x, y:
            if x == y:
                return 1
        case "<=", x, y:
            if x == y:
                return 1
        case "<", x, y:
            if x == y:
                return 0
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
    if not all(map(isConst, a[1:])):
        return a

    # if so, we can evaluate the term immediately
    a = tuple(map(unquote, a))
    match a:
        case "and", x, y:
            a = x and y
        case "or", x, y:
            a = x or y
        case "if", c, x, y:
            a = x if c else y
        case o, *s:
            f = interpreter.genv[o]
            a = f(*s)
        case _:
            raise Exception(a)
    return quote(a)


if __name__ == "__main__":
    assert simplify(("if", 2, 1, 0)) == 1
    print("ok")
