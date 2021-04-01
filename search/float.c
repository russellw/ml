#include "main.h"

// interned floating-point numbers
static si cap = 0x10;
static si count;
static Float **entries;

static size_t hash(Float *x) { return XXH64(&x->val,sizeof x->val,0); }

static si eq(Float *x, Float *y) { return !memcmp(&x->val, &y->val,sizeof x->val); }

static si slot(Float **entries, si cap, Float *x) {
  si mask = cap - 1;
  si i = hash(x) & mask;
  while (entries[i] && !eq(entries[i], x))
    i = (i + 1) & mask;
  return i;
}

static void expand(void) {
  assert(ispow2(cap));
  si cap1 = cap * 2;
  Float **entries1 = xcalloc(cap1, sizeof *entries);
  for (si i = 0; i < cap; i++) {
    Float *x = entries[i];
    if (x)
      entries1[slot(entries1, cap1, x)] = x;
  }
  free(entries);
  cap = cap1;
  entries = entries1;
}

static Float *store(Float *x) {
  Float *r = xmalloc(sizeof *x);
  *r = *x;
  return r;
}

void init_floats(void) { entries = xcalloc(cap, sizeof *entries); }

Float *intern_float(Float *x) {
  si i = slot(entries, cap, x);
  if (entries[i])
    return entries[i];
  if (++count > cap * 3 / 4) {
    expand();
    i = slot(entries, cap, x);
  }
  return entries[i] = store(x);
}
