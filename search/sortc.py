# Sort elements of C code
# Assumes clang-format already run
# Does not work for all possible programs!
# Test carefully before reusing in other projects

import argparse
import os
import re
import tempfile

parser = argparse.ArgumentParser(description="Sort elements of C code")
parser.add_argument("files", nargs="+")
args = parser.parse_args()


def chunkKey(chunk):
    if chunk:
        m = re.match(r"[\w ]+ \**(\w+)\(", chunk[0])
        if m:
            name = m[1]
            return [name] + chunk
    return [""] + chunk


def err(i, s):
    raise ValueError(f"{filename}:{i}: {s}")


def flatten(chunks):
    r = []
    for chunk in chunks:
        r.extend(chunk)
    return r


def getIndent(i):
    s = text[i]
    j = 0
    while j < len(s) and s[j] == " ":
        j += 1
    if j == len(s):
        return 1000
    if s[j] == "#":
        return 1000
    if s[j] == "\t":
        err(i, "tab detected")
    return j


def sortSpan(i, j):
    text[i:j] = sorted(text[i:j])


def sortSpans(spans):
    chunks = []
    for i, j in spans:
        chunk = text[i:j]
        while chunk and not chunk[-1]:
            chunk = chunk[:-1]
        if chunk:
            chunks.append(chunk)
    chunks.sort(key=chunkKey)
    i = spans[0][0]
    j = spans[-1][1]
    text[i:j] = flatten(chunks)


def sortSpansBlank(spans):
    chunks = []
    for i, j in spans:
        chunk = text[i:j]
        while chunk and not chunk[-1]:
            chunk = chunk[:-1]
        if chunk:
            chunks.append(chunk)
    chunks.sort(key=chunkKey)
    for i in range(0, len(chunks) - 1):
        chunks[i].append("")
    i = spans[0][0]
    j = spans[-1][1]
    text[i:j] = flatten(chunks)


# Cases


def casesSpan(i):
    while re.match(r"\s*case .*:", text[i]):
        i += 1
    return i


def sortCases():
    # Sort the order of cases within case blocks
    # This should be done before sorting the order of case blocks within switch statements
    i = 0
    while i < len(text):
        if re.match(r"\s*case .*:", text[i]):
            j = casesSpan(i)
            # The tricky bit is that the last case might end with {
            # which needs to be moved to the new last case
            if text[j - 1].endswith(" {"):
                text[j - 1] = text[j - 1][:-2]
                sortSpan(i, j)
                text[j - 1] += " {"
            else:
                sortSpan(i, j)
            i = j
            continue
        i += 1


# Switches


def caseBlockSpan(i):
    indent = getIndent(i)
    if not re.match(r"\s*(case .*|default):", text[i]):
        err(i, "bad case")
    while getIndent(i) == indent:
        i += 1
    if text[i - 1].endswith("{"):
        # Braced case block
        while getIndent(i) > indent:
            i += 1
        if getIndent(i) != indent:
            err(i, "bad indent")
        if not re.match(r"\s*}", text[i]):
            err("bad braces")
        i += 1
        if getIndent(i) != indent:
            err(i, "bad indent")
    else:
        # Unbraced case block
        while getIndent(i) > indent:
            i += 1
        if getIndent(i) != indent:
            err(i, "bad indent")
    return i


def caseBlockTerminated(i):
    i -= 1
    if re.match(r"\s*}", text[i]):
        i -= 1
    m = re.match(r"\s*(\w+)", text[i])
    if m:
        return m[1] in (
            "break",
            "continue",
            "err",
            "exit",
            "goto",
            "return",
            "unreachable",
        )


def caseBlockFallthruSpan(i):
    while re.match(r"\s*(case .*|default):", text[i]):
        i = caseBlockSpan(i)
        if caseBlockTerminated(i):
            return i
    return i


def caseBlockFallthruSpans(i):
    spans = []
    while re.match(r"\s*(case .*|default):", text[i]):
        j = caseBlockFallthruSpan(i)
        spans.append((i, j))
        i = j
    if not re.match(r"\s*}", text[i]):
        err(i, "bad braces")
    i += 1
    return spans, i


def sortSwitches(i, j):
    while i < j:
        if re.match(r"\s*switch \(.*\) {", text[i]):
            i += 1
            spans, i = caseBlockFallthruSpans(i)
            for i1, j1 in spans:
                sortSwitches(i1, j1)
            old = len(text)
            sortSpans(spans)
            i += len(text) - old
            continue
        i += 1


# Marked blocks


def blockSpan(i):
    if re.match(r"\s*}", text[i]):
        err(i, "bad block")
    if re.match(r"\s*// SORT$", text[i]):
        err(i, "nested SORT blocks not supported")
    indent = getIndent(i)
    i += 1
    while getIndent(i) > indent:
        if re.match(r"\s*// SORT$", text[i]):
            err(i, "nested SORT blocks not supported")
        i += 1
    if re.match(r"\s*}", text[i]):
        i += 1
    return i


def blockSpans(i):
    spans = []
    while not re.match(r"\s*///$", text[i]):
        j = blockSpan(i)
        spans.append((i, j))
        i = j
    i += 1
    return spans, i


def sortMarked():
    i = 0
    while i < len(text):
        if re.match(r"\s*// SORT$", text[i]):
            i += 1
            spans, i = blockSpans(i)
            old = len(text)
            for i1, j1 in spans:
                if text[i1].endswith("{"):
                    sortSpansBlank(spans)
                    break
            else:
                sortSpans(spans)
            i += len(text) - old
            continue
        i += 1


# Top level


def sortFile():
    global text

    # Skip files that are not C source files
    # The omission of other common extensions is intentional
    # This program should be carefully evaluated for required modification
    # before being reused in other projects
    if os.path.splitext(filename)[1] not in (".c", ".h"):
        return

    with open(filename) as f:
        text = [s.rstrip("\n") for s in f]
    old = text.copy()

    sortCases()
    sortSwitches(0, len(text))
    sortMarked()

    # Don't write unchanged files
    if text == old:
        return

    # Report which files we actually sorted
    print(filename)

    # Backup
    with open(tempfile.gettempdir() + "/" + os.path.split(filename)[1], "w") as f:
        for s in old:
            f.write(s + "\n")

    # Changed file
    with open(filename, "w") as f:
        for s in text:
            f.write(s + "\n")


for arg in args.files:
    if os.path.isfile(arg):
        filename = arg
        sortFile()
        continue
    for root, dirs, files in os.walk(arg):
        for filename in files:
            filename = os.path.join(root, filename)
            sortFile()
