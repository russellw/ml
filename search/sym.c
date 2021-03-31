#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

// interned symbols
static si cap = 0x100;
static si count;
static sym **entries;

static si eq(char *s, si n, char *s1, si n1) {
  if (n != n1)
    return 0;
  while (n--)
    if (*s++ != *s1++)
      return 0;
  return 1;
}

static si slot(sym **entries, si cap, char *s, si n) {
  si mask = cap - 1;
  si i = fnv(s, n) & mask;
  while (entries[i] && !eq(entries[i]->v, entries[i]->n, s, n))
    i = (i + 1) & mask;
  return i;
}

static void expand(void) {
  assert(ispow2(cap));
  si cap1 = cap * 2;
  sym **entries1 = xcalloc(cap1, sizeof *entries);
  for (si i = 0; i < cap; i++) {
    sym *x = entries[i];
    if (x)
      entries1[slot(entries1, cap1, x->v, x->n)] = x;
  }
  free(entries);
  cap = cap1;
  entries = entries1;
}

static sym *store(char *s, si n) {
  sym *r = mmalloc(offsetof(sym, v) + n);
  r->n = n;
  memcpy(r->v, s, n);
  return r;
}

void init_syms(void) {
  entries = xcalloc(cap, sizeof *entries);
  for (si i = 0; i < sizeof keywords / sizeof *keywords; i++) {
    sym *x = keywords + i;
    assert(strlen(x->v) < sizeof x->v);
    count++;
    assert(count <= cap * 3 / 4);
    si j = slot(entries, cap, x->v, strlen(x->v));
    assert(!entries[j]);
    entries[j] = x;
  }
}

sym *intern(char *p, si n) {
  si i = slot(entries, cap, p, n);
  if (entries[i])
    return entries[i];
  if (++count > cap * 3 / 4) {
    expand();
    i = slot(entries, cap, p, n);
  }
  return entries[i] = store(p, n);
}
