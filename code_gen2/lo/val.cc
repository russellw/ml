#include <olivine.h>

val** begin(val* a) {
	assert(a->tag == t_list);
	auto a1 = (list*)a;
	return a1->v;
}

val** end(val* a) {
	assert(a->tag == t_list);
	auto a1 = (list*)a;
	return a1->v + a1->n;
}

size_t kw(val* a) {
	assert(a->tag == t_list);
	auto a1 = (list*)a;
	return keyword(*a1->v);
}
