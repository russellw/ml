#include "olivine.h"

#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#define O_BINARY 0
#endif

char buf[bufsz];

// SORT
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

void readFile(const char* file, vector<char>& text) {
	auto f = open(file, O_BINARY | O_RDONLY);
	struct stat st;
	if (f < 0 || fstat(f, &st)) {
		perror(file);
		exit(1);
	}
	auto n = st.st_size;
	text.resize(n + 1);
	if (read(f, text.data(), n) != n) {
		perror(file);
		exit(1);
	}
	close(f);
	text[n] = 0;
}

void* xcalloc(size_t n, size_t size) {
	auto r = calloc(n, size);
	if (!r) {
		perror("calloc");
		exit(1);
	}
	return r;
}

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
///
