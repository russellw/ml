#include "main.h"

static void reserve(vec *v, si n) {
  if (n <= v->cap)
    return;
  v->cap = max(n, v->cap * 2);
  if (v->p == v->w) {
    v->p = xmalloc(v->cap * sizeof(si));
    memcpy(v->p, v->w, v->n * sizeof(si));
    return;
  }
  v->p = xrealloc(v->p, v->cap * sizeof(si));
}

static void resize(vec *v, si n) {
  reserve(v, n);
  v->n = n;
}

void vpush(vec *v, si a) {
  reserve(v, v->n + 1);
  v->p[v->n++] = a;
}

void vfree(vec *v) {
  if (v->p != v->w)
    free(v->p);
}
