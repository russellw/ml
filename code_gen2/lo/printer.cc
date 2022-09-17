#include <olivine.h>

namespace {
void name(dyn a) {
	print(a);
}

void decl(dyn f) {
	assert(f.kw() == s_fn);
	name(f[1]);
	putchar(' ');
	name(f[2]);
	putchar('(');
	bool more = 0;
	for (auto x: f[3]) {
		if (more) putchar(',');
		more = 1;
	}
	putchar(')');
}
} // namespace

void printcc(dyn program) {
	puts("#include <olivine.h>");

	//function declarations
	for (auto a: program)
		if (a.kw() == s_fn) {
			decl(a);
			puts(";");
		}

	//function definitions
	for (auto a: program)
		if (a.kw() == s_fn) {
			decl(a);
			puts("{");
			puts("}");
		}
}
