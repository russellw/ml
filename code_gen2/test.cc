#include <olivine.h>

void main() {
	dyn a(1.0);
	assert(a.tag() == d_num);

	puts("ok");
}
