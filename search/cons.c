#include "main.h"

// interned cons lists
static si cap = 0x10000;
static si count;
static Cons **entries;

static size_t hash(Cons *x) { return XXH64(x, sizeof *x, 0); }

static int eq(Cons *x, Cons *y) { return x->hd == y->hd && x->tl == y->tl; }

static si slot(Cons **entries, si cap, Cons *x) {
  si mask = cap - 1;
  si i = hash(x) & mask;
  while (entries[i] && !eq(entries[i], x))
    i = (i + 1) & mask;
  return i;
}

static void expand(void) {
  assert(ispow2(cap));
  si cap1 = cap * 2;
  Cons **entries1 = xcalloc(cap1, sizeof *entries);
  for (si i = 0; i < cap; i++) {
    Cons *x = entries[i];
    if (x)
      entries1[slot(entries1, cap1, x)] = x;
  }
  free(entries);
  cap = cap1;
  entries = entries1;
}

static Cons *store(Cons *x) {
  Cons *r = xmalloc(sizeof *x);
  *r = *x;
  return r;
}

void init_cons(void) { entries = xcalloc(cap, sizeof *entries); }

static si listr(si *p, si *end) {
  if (p == end)
    return nil;
  return cons(*p, listr(p + 1, end));
}

// SORT
si cons(si hd, si tl) {
  if (tag(tl) != t_cons)
    err("cons: not a list");
  Cons x;
  x.hd = hd;
  x.tl = tl;
  si i = slot(entries, cap, &x);
  if (!entries[i]) {
    if (++count > cap * 3 / 4) {
      expand();
      i = slot(entries, cap, &x);
    }
    entries[i] = store(&x);
  }
  return term(entries[i], t_cons);
}

si get(si env, si key, int *found) {
  for (; env != nil; env = tl(env)) {
    si record = hd(env);
    switch (keyword(hd(record))) {
    case w_let:
      record = tl(record);
      if (hd(record) == key) {
        *found = 1;
        record = tl(record);
        return hd(record);
      }
      break;
    case w_letrec:
      record = tl(record);
      while (record != nil) {
        si key1 = hd(record);
        record = tl(record);
        si params = hd(record);
        record = tl(record);
        si body = hd(record);
        if (key1 == key) {
          *found = 1;
          return list3(env, params, body);
        }
        record = tl(record);
      }
      break;
    }
  }
  *found = 0;
  return nil;
}

si hd(si s) {
  if (tag(s) != t_cons)
    return s;
  if (s == nil)
    return nil;
  return consp(s)->hd;
}

si list(vec *v) {
  si r = listr(v->p, v->p + v->n);
  vfree(v);
  return r;
}

si list1(si a) { return cons(a, nil); }

si list2(si a, si b) { return cons(a, list1(b)); }

si list3(si a, si b, si c) { return cons(a, list2(b, c)); }

si list4(si a, si b, si c, si d) { return cons(a, list3(b, c, d)); }

si tl(si s) {
  if (tag(s) != t_cons)
    return nil;
  if (s == nil)
    return nil;
  return consp(s)->tl;
}
///
