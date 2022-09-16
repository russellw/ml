#include <olivine.h>

char keywords[][16] = {
#define _(x) #x,
#include <lo/keywords.h>
};

namespace {
// Compare a null terminated string with a counted one.
bool eq(const char* z, const char* s, size_t n) {
	while (n--)
		if (*z++ != *s++) return 0;
	return !*z;
}

size_t slot(char** entries, size_t cap, const char* s, size_t n) {
	size_t mask = cap - 1;
	auto i = fnv(s, n) & mask;
	while (entries[i] && !eq(entries[i], s, n)) i = (i + 1) & mask;
	return i;
}

size_t cap = 0x100;
size_t qty = end_s;
char** entries;

struct init {
	init() {
		static_assert(isPow2(sizeof(*keywords)));
		assert(isPow2(cap));
		assert(qty <= cap * 3 / 4);
		entries = (char**)xcalloc(cap, sizeof(char*));
		for (int i = 0; i != sizeof keywords / sizeof *keywords; ++i) {
			auto s = keywords + i;
			auto n = strlen(s);

			// C++ allows the edge case where a string literal exactly fills an array, leaving no room for a null terminator. This
			// is sometimes useful, but would not be appropriate here, so make sure it's not the case.
			assert(n < sizeof *keywords);

			// Make sure there are no duplicate keywords.
			assert(!entries[slot(entries, cap, s, n)]);

			// Add to hash table.
			entries[slot(entries, cap, s, n)] = s;
		}
	}
} _;

void expand() {
	auto cap1 = cap * 2;
	auto entries1 = (char**)xcalloc(cap1, sizeof(char*));
	for (size_t i = 0; i != cap; ++i) {
		auto s = entries[i];
		if (s) entries1[slot(entries1, cap1, s, strlen(s))] = s;
	}
	free(entries);
	cap = cap1;
	entries = entries1;
}
} // namespace

char* intern(const char* s, size_t n) {
	auto i = slot(entries, cap, s, n);

	// If we have seen this string before, return the existing one
	if (entries[i]) return entries[i];

	// Expand the hash table if necessary.
	if (++qty > cap * 3 / 4) {
		expand();
		i = slot(entries, cap, s, n);
		assert(!entries[i]);
	}

	// Make a new symbol.
	auto r = new char[n + 1];
	memcpy(r, s, n);
	r[n] = 0;

	// Add to hash table.
	return entries[i] = r;
}
