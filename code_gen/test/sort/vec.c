#include "sort.h"

void reserve(struct Vec* v, int n) {
	if (n <= v->cap) return;
	v->cap = max(n, v->cap * 2);
	v->p = realloc(v->p, v->cap * sizeof(void*));
	if (!v->p) {
		perror("realloc");
		exit(1);
	}
}

__declspec(dllexport) void push(struct Vec* v, void* a) {
	reserve(v, v->n + 1);
	v->p[v->n++] = a;
}
