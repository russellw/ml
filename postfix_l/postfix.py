from etc import *


def compose(a):
    s = []

    def rec(a):
        if isinstance(a, str):
            s.append(a)
            return
        if isinstance(a, tuple):
            s.append("(")
            for b in a:
                rec(b)
            s.append(")")
            return
        if isinstance(a, float) or isinstance(a, int):
            s.append("{")
            a = str(a)
            if a.endswith(".0"):
                a = a[:-2]
            for c in a:
                s.append(c)
            s.append("}")
            return
        raise TypeError()

    rec(a)
    return s


if __name__ == "__main__":
    assert compose(3.0) == ["{", "3", "}"]
    assert compose(("+", 3, "x0")) == ["(", "+", "{", "3", "}", "x0", ")"]

    print("ok")
