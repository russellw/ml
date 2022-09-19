#include <olivine.h>

int main() {
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
		dyn foo("foo");
		assert(foo.isSym());

		dyn a("a");
		assert(a.isSym());
		assert(!strcmp(a.str(), "a"));

		dyn a1("a");
		assert(a == a1);
		assert(a != foo);

		a = dyn("if");
		assert(a.kw() == s_if);

		a = dyn("return");
		assert(a.kw() == s_return);

		a = dyn("qwertyuiop");
		assert(a.kw() >= end_s);
	}

	//numbers
	{
		dyn a(1.0);
		assert(a.isNum());
		assert(a.num() == 1.0);
	}

	//lists
	{
		auto x = dyn(1.0);
		auto y = dyn(2.0);
		auto z = dyn(3.0);
		assert(x != y);

		auto a = list(x);
		assert(a.size() == 1);
		assert(a[0] == x);

		a = list(x, y);
		assert(a.size() == 2);
		assert(a[0] == x);
		assert(a[1] == y);

		vector<dyn> v;
		v.push_back(x);
		v.push_back(y);
		a = list(v);
		assert(a.size() == 2);
		assert(a[0] == x);
		assert(a[1] == y);

		size_t n = 0;
		for (auto b: a) n+=b.size();
		assert(n == 2);

		assert(a.kw() != s_fn);

		a = list(sym(s_fn), y);
		assert(a.kw() == s_fn);

		assert(x != list(x, x));
		assert(list(x, x) == list(x, x));
		assert(list(x, x) != list(x, y));

		assert(list(x, y, z).from(0) == list(x, y, z));
		assert(list(x, y, z).from(1) == list(y, z));
		assert(list(x, y, z).from(2) == list(z));
		assert(list(x, y, z).from(3) == list());
		assert(list(x, y, z).from(4) == list());
	}

	//program output
	vector<dyn> program;
	vector<dyn> f;
	vector<dyn> params;

	f.clear();
	f.push_back(sym(s_fn));
	f.push_back(dyn("int"));
	f.push_back(dyn("square"));
	params.clear();
	params.push_back(list(dyn("int"), dyn("x")));
	f.push_back(list(params));
	f.push_back(list(s_goto, dyn("foo")));
	f.push_back(list(s_label, dyn("foo")));
	f.push_back(list(dyn("return"), list(s_mul, dyn("x"), dyn("x"))));
	program.push_back(list(f));

	f.clear();
	f.push_back(sym(s_fn));
	f.push_back(dyn("int"));
	f.push_back(dyn("factorial"));
	params.clear();
	params.push_back(list(dyn("int"), dyn("n")));
	f.push_back(list(params));
	f.push_back(list(
		s_if,
		list(s_le, dyn("n"), dyn(1.0)),
		list(s_return, dyn(1.0)),
		list(s_return, list(s_mul, dyn("n"), list(dyn("factorial"), list(s_sub, dyn("n"), dyn(1.0)))))));
	program.push_back(list(f));

	f.clear();
	f.push_back(sym(s_fn));
	f.push_back(dyn("int"));
	f.push_back(dyn("main"));
	params.clear();
	f.push_back(list(params));
	f.push_back(list(dyn("assert"), dyn(1.0)));
	f.push_back(list(dyn("assert"), list(s_eq, dyn(1.0), dyn(1.0))));
	f.push_back(list(dyn("assert"), list(s_eq, list(dyn("square"), dyn(3.0)), dyn(9.0))));
	f.push_back(list(dyn("assert"), list(s_eq, list(dyn("factorial"), dyn(5.0)), dyn(120.0))));
	f.push_back(list(dyn("return"), list(s_sub, dyn(1.0), dyn(1.0))));
	program.push_back(list(f));

	printcc(list(program));
	return 0;
}
