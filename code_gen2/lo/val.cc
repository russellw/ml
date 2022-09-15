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
