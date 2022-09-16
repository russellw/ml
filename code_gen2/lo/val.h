enum
{
	t_list,
	t_num,
	t_sym,
};

struct val {
	unsigned char tag;

	val(int tag): tag(tag) {
	}
};

typedef val* dyn;

dyn* begin(dyn a);
dyn* end(dyn a);

size_t kw(dyn a);
void print(dyn a);
dyn at(dyn a, size_t i);
