#include "main.h"

// interned rationals
static si cap = 0x10;
static si count;
static Rat **entries;

static size_t hash(Rat *x) {
  return mpz_get_ui(mpq_numref(x->val)) ^ mpz_get_ui(mpq_denref(x->val));
}

static si eq(Rat *x, Rat *y) { return mpq_equal(x->val, y->val); }

static si slot(Rat **entries, si cap, Rat *x) {
  si mask = cap - 1;
  si i = hash(x) & mask;
  while (entries[i] && !eq(entries[i], x))
    i = (i + 1) & mask;
  return i;
}

static void expand(void) {
  assert(ispow2(cap));
  si cap1 = cap * 2;
  Rat **entries1 = xcalloc(cap1, sizeof *entries);
  for (si i = 0; i < cap; i++) {
    Rat *x = entries[i];
    if (x)
      entries1[slot(entries1, cap1, x)] = x;
  }
  free(entries);
  cap = cap1;
  entries = entries1;
}

static Rat *store(Rat *x) {
  Rat *r = xmalloc(sizeof *x);
  *r = *x;
  return r;
}

void init_rats(void) { entries = xcalloc(cap, sizeof *entries); }

si irat(Rat *x) {
  mpq_canonicalize(x->val);
  si i = slot(entries, cap, x);
  if (entries[i])
    mpq_clear(x->val);
  else {
    if (++count > cap * 3 / 4) {
      expand();
      i = slot(entries, cap, x);
    }
    entries[i] = store(x);
  }
  return term(entries[i], t_rat);
}
