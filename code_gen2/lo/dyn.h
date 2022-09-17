enum
{
	t_list,
	t_num,
	t_sym,
};

class dyn {
	size_t x;

public:
	dyn(void* p, size_t tag): x(size_t(p) + tag) {
	}

	explicit dyn(const char* s): x(size_t(p) + t_sym) {
	}

	dyn* begin();
	dyn* end();

	size_t kw() const;
	void* ptr() const {
		return (void*)(x & ~size_t(3));
	}
};

dyn list(dyn a);
dyn list(dyn a, dyn b);
dyn list(const vector<dyn>& v);

void print(dyn a);
