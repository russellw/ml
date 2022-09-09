#include "sort.h"

__declspec(dllimport) void readFile(char* file, struct Vec* v);
__declspec(dllimport) void print(struct Str* s);

int cmp(const void* ap, const void* bp) {
	struct Str* a = *(struct Str**)ap;
	struct Str* b = *(struct Str**)bp;
	int n = min(a->n, b->n);
	int c = memcmp(a->v, b->v, n);
	if (c) return c;
	return a->n - b->n;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "Usage: sort <file>\n");
		exit(1);
	}
	struct Vec v = {0};
	readFile(argv[1], &v);
	qsort(v.p, v.n, sizeof(void*), cmp);
	for (int i = 0; i < v.n; i++) {
		print(v.p[i]);
		putchar('\n');
	}
	return 0;
}
