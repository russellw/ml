#include <olivine.h>

void* xmalloc(size_t bytes) {
	auto r = malloc(bytes);
	if (!r) {
		perror("malloc");
		exit(1);
	}
#ifdef DEBUG
	memset(r, 0xcc, bytes);
#endif
	return r;
}

void* xcalloc(size_t n, size_t size) {
	auto r = calloc(n, size);
	if (!r) {
		perror("calloc");
		exit(1);
	}
	return r;
}

size_t fnv(const void* p, size_t bytes) {
	// Fowler-Noll-Vo-1a is slower than more sophisticated hash algorithms for large chunks of data, but faster for tiny ones, so it
	// still sees use.
	auto q = (const unsigned char*)p;
	size_t h = 2166136261u;
	while (bytes--) {
		h ^= *q++;
		h *= 16777619;
	}
	return h;
}
