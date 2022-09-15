struct list: val {
	unsigned n;
	val* v[];
};

list* mk(val* a);
list* mk(val* a, val* b);
list* mk(const vector<val*>& v);
