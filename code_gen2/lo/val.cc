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

void print(val* a) {
	switch (a->tag) {
	case t_list:
		print((list*)a);
		break;
	case t_sym:
		print((sym*)a);
		break;
	case t_num:
		print((num*)a);
		break;
	default:
		unreachable;
	}
}
