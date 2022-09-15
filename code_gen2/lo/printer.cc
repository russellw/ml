#include <olivine.h>

namespace {}

void printcc(val* program) {
	puts("#include <olivine.h>");

	//function declarations
	for (auto a: program)
		if (kw(a) == s_fn) {}
}
