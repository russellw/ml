#include "sort.h"

__declspec(dllexport) struct Str* str(char* s) {
	int n = strlen(s);
	struct Str* r = malloc(offsetof(struct Str, v) + n);
	if (!r) {
		perror("malloc");
		exit(1);
	}
	r->n = n;
	memcpy(r->v, s, n);
	return r;
}

__declspec(dllexport) void print(struct Str* s) {
	for (int i = 0; i < s->n; i++) putchar(s->v[i]);
}
