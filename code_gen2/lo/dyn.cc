#include <olivine.h>

struct List {
	size_t n;
	dyn v[];
};

List empty;

dyn::dyn(double a): x(size_t(s) + t_sym) {
	auto p = new double;
	*p = a;
	x = size_t(p) + t_num;
}

static List* list(int n) {
	auto r = (List*)xmalloc(offsetof(List, v) + n * sizeof(dyn));
	r->tag = t_list;
	r->n = n;
	return r;
}

List* list(dyn a) {
	auto r = list(1);
	r->v[0] = a;
	return r;
}

List* list(dyn a, dyn b) {
	auto r = list(2);
	r->v[0] = a;
	r->v[1] = b;
	return r;
}

List* list(const vector<dyn>& v) {
	auto r = list(v.size());
	memcpy(r->v, v.data(), v.size() * sizeof(dyn));
	return r;
}

void print(List* a) {
	putchar('(');
	bool more = 0;
	for (auto b: a) {
		if (more) putchar(' ');
		more = 1;
		print(b);
	}
	putchar(')');
}

dyn* begin(dyn a) {
	assert(a->tag == t_list);
	auto a1 = (List*)a;
	return a1->v;
}

dyn* end(dyn a) {
	assert(a->tag == t_list);
	auto a1 = (List*)a;
	return a1->v + a1->n;
}

size_t kw(dyn a) {
	assert(a->tag == t_list);
	auto a1 = (List*)a;
	return keyword(*a1->v);
}

void print(dyn a) {
	switch (a->tag) {
	case t_list:
		print((List*)a);
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
	auto a1 = (List*)a;
	assert(i < a1->n);
	return a1->v[i];
}

const char* dyn::str() const {
	assert(tag() == t_str);
	return (const char*)(x - t_str);
}

bool dyn::operator==(dyn b) const {
	if (x == b.x) return 1;
	if (tag() != b.tag()) return 0;
	switch (tag()) {
	case t_num:
		return *(double*)ptr() == *(double*)ptr();
	}
	return 0;
}

size_t dyn::size() const {
	assert(tag() == t_list);
	auto p = (list*)(x - t_list);
	return p->n;
}

dyn dyn::operator[](size_t i) const {
	assert(tag() == t_list);
	auto p = (list*)(x - t_list);
	assert(i < p->n);
	return p->v[i];
}
