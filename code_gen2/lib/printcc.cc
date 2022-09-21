#include "olivine.h"

namespace {
void pname(dyn a) { print(a); }

void ptype(dyn type, dyn name) {
	print(type);
	putchar(' ');
	pname(name);
}

void decl(dyn f) {
	assert(f.kw() == s_fn);
	ptype(f[1], f[2]);
	putchar('(');
	bool more = 0;
	for (auto x: f[3]) {
		if (more) putchar(',');
		more = 1;
		ptype(x[0], x[1]);
	}
	putchar(')');
}

void expr(dyn a) {
	if (a.size() < 2) {
		print(a);
		return;
	}
	switch (a.kw()) {
	case s_add:
	case s_div:
	case s_eq:
	case s_le:
	case s_lt:
	case s_mul:
	case s_ne:
	case s_rem:
	case s_sub:
		expr(a[1]);
		print(a[0]);
		expr(a[2]);
		break;
	default:
		pname(a[0]);
		putchar('(');
		bool more = 0;
		for (auto b: a.from(1)) {
			if (more) putchar(',');
			more = 1;
			expr(b);
		}
		putchar(')');
		break;
	}
}

void stmt(dyn a) {
	switch (a.kw()) {
	case s_goto:
	case s_return:
		print(a[0]);
		if (a.size() > 1) {
			putchar(' ');
			expr(a[1]);
		}
		break;
	case s_if:
		print(a[0]);
		putchar('(');
		expr(a[1]);
		puts(")");
		stmt(a[2]);
		if (a.size() > 3) {
			puts("else");
			stmt(a[3]);
		}
		break;
	case s_label:
		expr(a[1]);
		puts(":");
		return;
	default:
		expr(a);
		break;
	}
	puts(";");
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
			for (auto b: a.from(4)) stmt(b);
			puts("}");
		}
}
