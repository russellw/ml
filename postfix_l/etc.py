import inspect
import time

trace = 0


def tron():
    global trace
    trace = 1


def troff():
    global trace
    trace = 0


def printTime():
    print(time.strftime("%A, %B %d, %Y, %H:%M:%S", time.localtime()))


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


def fixLen(s, n, a=0):
    return s[:n] + (a,) * (n - len(s))


if __name__ == "__main__":
    assert fixLen([1, 2, 3], 5) == [1, 2, 3, 0, 0]
    assert fixLen([1, 2, 3], 2) == [1, 2]

    print("ok")
