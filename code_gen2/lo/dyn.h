enum
{
	t_list,
	t_num,
	t_sym,
};

class dyn {
	size_t x;

	size_t tag() const {
		return x & 3;
	}

public:
	//construct
	dyn(void* p, size_t tag): x(size_t(p) + tag) {
	}
	explicit dyn(const char* s);
	explicit dyn(double a);

	//classify
	bool isSym() const {
		return tag() == t_sym;
	}
	bool isNum() const {
		return tag() == t_num;
	}

	//extract
	const char* str() const {
		assert(isSym());
		return (const char*)(x - t_sym);
	}
	double num() const {
		assert(isNum());
		return *((double*)(x - t_num));
	}

	//compare
	bool operator==(dyn b) const;
	bool operator!=(dyn b) const {
		return !(*this == b);
	}

	//iterate
	dyn* begin() const;
	dyn* end() const;

	//etc
	size_t kw() const;
	size_t size() const;
	dyn operator[](size_t i) const;
	dyn from(size_t i) const;
};

inline dyn sym(int k) {
	return dyn(keywords[k], t_sym);
}

dyn list();
dyn list(dyn a);
dyn list(dyn a, dyn b);
dyn list(dyn a, dyn b, dyn c);
dyn list(const vector<dyn>& v);

dyn list(int a, dyn b);
dyn list(int a, dyn b, dyn c);
dyn list(int a, dyn b, dyn c, dyn d);

void print(dyn a);
