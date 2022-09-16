#include <olivine.h>

namespace {
void name(val* a) {
	print(a);
}

void decl(val* f) {
	assert(kw(f) == s_fn);
	name(at(f, 2));
	putchar('(');
	bool more = 0;
	for (auto x: at(f, 3)) {
		if (more) putchar(',');
		more = 1;
	}
	putchar(')');
}
} // namespace

void printcc(val* program) {
	puts("#include <olivine.h>");

	//function declarations
	for (auto a: program)
		if (kw(a) == s_fn) {
			decl(a);
			puts(";");
		}

	//function definitions
	for (auto a: program)
		if (kw(a) == s_fn) {
			decl(a);
			puts("{");
			puts("}");
		}
}
