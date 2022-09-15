struct num: val {
	double x;

	explicit num(double x): val(t_num), x(x) {
	}
};

void print(num* a);
