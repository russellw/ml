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
