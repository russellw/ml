import inspect
import time

trace = 0


def tron():
    global trace
    trace = 1


def troff():
    global trace
    trace = 0


def fname(i):
    return chr(ord("a") + i)


def printTime():
    print(time.strftime("%A, %B %d, %Y, %H:%M:%S", time.localtime()))


def compose(a):
    s = []

    def rec(a):
        if not isinstance(a, tuple):
            s.append(a)
            return
        s.append("(")
        for b in a:
            rec(b)
        s.append(")")

    rec(a)
    return s


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


def fixLen(s, n, a=0):
    return s[:n] + (a,) * (n - len(s))


if __name__ == "__main__":
    assert compose(3) == [3]
    assert compose(("+", 3, "x0")) == ["(", "+", 3, "x0", ")"]

    assert fixLen([1, 2, 3], 5) == [1, 2, 3, 0, 0]
    assert fixLen([1, 2, 3], 2) == [1, 2]

    print("ok")
