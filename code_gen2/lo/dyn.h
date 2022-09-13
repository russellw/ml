enum
{
	d_list,
	d_num,
	d_sym,
};

class dyn {
	size_t x;

	static size_t pack(void* p, size_t tag) {
		return size_t(p) + tag;
	}

public:
	explicit dyn(double val): x(pack(new double(val), d_num)) {
	}

	size_t tag() const {
		return x & 3;
	}
};
