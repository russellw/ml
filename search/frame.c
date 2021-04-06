#include "main.h"

// interned frames
static si cap = 0x1000;
static si count;
static frame **entries;

static size_t hash(frame *x) { return XXH64(x, sizeof *x, 0); }

static int eq(frame *x, frame *y) {
  return x->key == y->key && x->val == y->val && x->next == y->next;
}

static si slot(frame **entries, si cap, frame *x) {
  si mask = cap - 1;
  si i = hash(x) & mask;
  while (entries[i] && !eq(entries[i], x))
    i = (i + 1) & mask;
  return i;
}

static void expand(void) {
  assert(ispow2(cap));
  si cap1 = cap * 2;
  frame **entries1 = xcalloc(cap1, sizeof *entries);
  for (si i = 0; i < cap; i++) {
    frame *x = entries[i];
    if (x)
      entries1[slot(entries1, cap1, x)] = x;
  }
  free(entries);
  cap = cap1;
  entries = entries1;
}

static frame *store(frame *x) {
  frame *r = xmalloc(sizeof *x);
  *r = *x;
  return r;
}

void init_frames(void) { entries = xcalloc(cap, sizeof *entries); }

// SORT
si put(si fm, si key, si val) {
  if (tag(fm) != t_frame)
    err("put: not a frame");
  frame x;
  x.key = key;
  x.val = val;
  x.next = framep(fm);
  si i = slot(entries, cap, &x);
  if (!entries[i]) {
    if (++count > cap * 3 / 4) {
      expand();
      i = slot(entries, cap, &x);
    }
    entries[i] = store(&x);
  }
  return term(entries[i], t_frame);
}
///
