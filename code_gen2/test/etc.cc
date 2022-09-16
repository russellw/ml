#include <olivine.h>

void main() {
	assert(isPow2(1));
	assert(isPow2(2));
	assert(isPow2(4));
	assert(isPow2(8));
	assert(!isPow2(3));
	assert(!isPow2(5));
	assert(!isPow2(6));
	assert(!isPow2(7));
	assert(!isPow2(9));
	assert(!isPow2(10));

	assert(fnv("", 0) == fnv("", 0));
	assert(fnv("abc", 3) == fnv("abc", 3));

	//symbols
	{
		sym foo("foo");
		assert(foo.tag == t_sym);

		auto a = intern("a");
		assert(a->tag == t_sym);
		assert(!strcmp(a->z, "a"));

		auto a1 = intern("a");
		assert(a == a1);

		a = intern("if");
		assert(keyword(a) == s_if);

		a = intern("return");
		assert(keyword(a) == s_return);

		a = intern("qwertyuiop");
		assert(keyword(a) >= end_s);
	}

	//numbers
	{
		auto a = new num(1.0);
		assert(a->tag == t_num);
		assert(a->x == 1.0);
	}

	//lists
	{
		auto x = new num(1.0);
		auto y = new num(2.0);

		auto a = mk(x);
		assert(a->tag == t_list);
		assert(a->n == 1);
		assert(a->v[0] == x);

		a = mk(x, y);
		assert(a->tag == t_list);
		assert(a->n == 2);
		assert(a->v[0] == x);
		assert(a->v[1] == y);

		vector<val*> v;
		v.push_back(x);
		v.push_back(y);
		a = mk(v);
		assert(a->tag == t_list);
		assert(a->n == 2);
		assert(a->v[0] == x);
		assert(a->v[1] == y);

		int n = 0;
		for (auto b: a) ++n;
		assert(n == 2);

		assert(kw(a) != s_fn);

		a = mk(intern("fn"), y);
		assert(kw(a) == s_fn);
	}

	//program output
	vector<val*> program;
	vector<val*> f;

	f.clear();
	f.push_back(intern("fn"));
	f.push_back(intern("void"));
	f.push_back(intern("main"));
	f.push_back(&empty);
	program.push_back(mk(f));

	printcc(mk(program));
}
