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
	dyn(void* p, size_t tag): x(size_t(p) + tag) {
	}

	explicit dyn(const char* s): x(size_t(s) + t_sym) {
	}

	explicit dyn(double a);

	dyn* begin();
	dyn* end();

	size_t kw() const;
	size_t size() const;
	void* ptr() const {
		return (void*)(x & ~size_t(3));
	}
	double num() const {
		assert(isNum());
		return *((double*)(x - t_num));
	}
	bool isSym() const {
		return tag() == t_sym;
	}
	bool isNum() const {
		return tag() == t_num;
	}
	const char* str() const;
	bool operator==(dyn b) const;
	dyn operator[](size_t i) const;
};

dyn list(dyn a);
dyn list(dyn a, dyn b);
dyn list(const vector<dyn>& v);

void print(dyn a);
