struct list: val {
	unsigned n;
	val* v[];

	list(): val(t_list) {
	}
};

extern list empty;

list* mk(val* a);
list* mk(val* a, val* b);
list* mk(const vector<val*>& v);

inline val** begin(list* a) {
	return a->v;
}

inline val** end(list* a) {
	return a->v + a->n;
}

void print(list* a);
