// Keywords are strings that are known to be important.
enum
{
#define k(x) s_##x,
#define o(x, s) s_##x,
#include <lo/keywords.h>
	end_s
};

extern char keywords[][16];

inline size_t keyword(const void* p) {
	// Assign the difference to an unsigned variable and perform the division explicitly, because ptrdiff_t is a signed type, but
	// unsigned division is slightly faster.
	size_t i = (char*)p - *keywords;
	return i / sizeof *keywords;
}

char* intern(const char* s, size_t n);

inline char* intern(const char* s) {
	return intern(s, strlen(s));
}
