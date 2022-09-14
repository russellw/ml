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

	sym foo("foo");
	assert(foo.tag == t_sym);

	puts("ok");
}
