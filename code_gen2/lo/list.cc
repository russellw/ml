#include <olivine.h>

list empty;

static list* mk(int n) {
	auto r = (list*)xmalloc(offsetof(list, v) + n * sizeof(dyn));
	r->tag = t_list;
	r->n = n;
	return r;
}

list* mk(dyn a) {
	auto r = mk(1);
	r->v[0] = a;
	return r;
}

list* mk(dyn a, dyn b) {
	auto r = mk(2);
	r->v[0] = a;
	r->v[1] = b;
	return r;
}

list* mk(const vector<dyn>& v) {
	auto r = mk(v.size());
	memcpy(r->v, v.data(), v.size() * sizeof(dyn));
	return r;
}

void print(list* a) {
	putchar('(');
	bool more = 0;
	for (auto b: a) {
		if (more) putchar(' ');
		more = 1;
		print(b);
	}
	putchar(')');
}
