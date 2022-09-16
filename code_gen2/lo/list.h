struct list: val {
	unsigned n;
	dyn v[];

	list(): val(t_list) {
	}
};

extern list empty;

list* mk(dyn a);
list* mk(dyn a, dyn b);
list* mk(const vector<dyn>& v);

inline dyn* begin(list* a) {
	return a->v;
}

inline dyn* end(list* a) {
	return a->v + a->n;
}

void print(list* a);
