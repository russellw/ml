#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

// uninterned symbols
namespace {
pool<> uninterned;

si roundup(si n) { return n + 7 & ~(si)7; }
} // namespace

sym *mkSym(type t) {
  // uninterned symbols may later be given names, so must allocate enough memory
  // to hold the name; the amount needed depends on the naming pattern, which is
  // e.g. sK followed by an integer; 64-bit integers can go up to 19 digits, so
  // rounding the required space up to the nearest 8 bytes gives 24
  auto r = (sym *)uninterned.alloc(roundup(offsetof(sym, v) + 24));
  r->thisType = type::none;
  r->t = t;
  *r->v = 0;
  return r;
}

// interned symbols
namespace {
// must be a power of 2, and large enough to hold the largest collection of
// entries that will be loaded at initialization time
si cap = 0x100;
si count;
sym **entries = (sym **)xcalloc(cap, sizeof *entries);

bool eq(const char *s, const char *p, si n) {
  while (n--)
    if (*s++ != *p++)
      return 0;
  return !*s;
}

si slot(sym **entries, si cap, const char *p, si n) {
  auto mask = cap - 1;
  auto i = fnv(p, n) & mask;
  while (entries[i] && !eq(entries[i]->v, p, n))
    i = (i + 1) & mask;
  return i;
}

void expand() {
  assert(isPow2(cap));
  auto cap1 = cap * 2;
  auto entries1 = (sym **)xcalloc(cap1, sizeof *entries);
  for (auto i = entries, e = entries + cap; i != e; ++i) {
    auto s = *i;
    if (s)
      entries1[slot(entries1, cap1, s->v, strlen(s->v))] = s;
  }
  free(entries);
  cap = cap1;
  entries = entries1;
}

struct init {
  init() {
    for (si i = 0; i != sizeof keywords / sizeof *keywords; ++i) {
      auto s = keywords + i;
      assert(strlen(s->v) < sizeof s->v);
      ++count;
      assert(count <= cap * 3 / 4);
      auto j = slot(entries, cap, s->v, strlen(s->v));
      assert(!entries[j]);
      entries[j] = s;
    }
  }
} init1;

sym *store(const char *s, si n) {
  auto r = (sym *)mmalloc(offsetof(sym, v) + n + 1);
  memset(r, 0, offsetof(sym, v));
  memcpy(r->v, s, n);
  r->v[n] = 0;
  return r;
}

sym *put(const char *p, si n) {
  auto i = slot(entries, cap, p, n);
  if (entries[i])
    return entries[i];
  if (++count > cap * 3 / 4) {
    expand();
    i = slot(entries, cap, p, n);
  }
  return entries[i] = store(p, n);
}
} // namespace

sym *intern(const char *s, si n) { return put(s, n); }

void initSyms() {
  uninterned.init();
  for (auto i = entries, e = entries + cap; i != e; ++i) {
    auto s = *i;
    if (s)
      s->t = type::none;
  }
}

#ifdef DEBUG
void ck(const sym *s) {
  ckPtr(s);
  ckStr(s->v);
  ck(s->thisType);
  ck(s->t);
}
#endif
