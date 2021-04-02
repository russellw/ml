#include "main.h"

static void reserve(vec *v, si m) {
  if (m <= v->cap)
    return;
  v->cap = max(m, v->cap * 2);
  if (v->p == v->w) {
    v->p = xmalloc(v->cap * sizeof(si));
    memcpy(v->p, v->w, v->n * sizeof(si));
    return;
  }
  v->p = xrealloc(v->p, v->cap * sizeof(si));
}

static void resize(vec *v, si m) {
  reserve(v, m);
  v->n = m;
}

void vpush(vec *v, si a) {
  reserve(v, v->n + 1);
  v->p[v->n++] = a;
}

void vfree(vec *v) {
  if (v->p != v->w)
    free(v->p);
}
