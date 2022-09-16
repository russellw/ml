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

	size_t kw() const;
	void* ptr() const {
		return (void*)(x & ~size_t(3));
	}
};

dyn* begin(dyn a);
dyn* end(dyn a);

size_t kw(dyn a);
void print(dyn a);
dyn at(dyn a, size_t i);
