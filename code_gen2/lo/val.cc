#include <olivine.h>

dyn* begin(dyn a) {
	assert(a->tag == t_list);
	auto a1 = (list*)a;
	return a1->v;
}

dyn* end(dyn a) {
	assert(a->tag == t_list);
	auto a1 = (list*)a;
	return a1->v + a1->n;
}

size_t kw(dyn a) {
	assert(a->tag == t_list);
	auto a1 = (list*)a;
	return keyword(*a1->v);
}

void print(dyn a) {
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

dyn at(dyn a, size_t i) {
	assert(a->tag == t_list);
	auto a1 = (list*)a;
	assert(i < a1->n);
	return a1->v[i];
}
