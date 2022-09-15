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

val** begin(val* a);
val** end(val* a);

size_t kw(val* a);
