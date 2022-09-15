// Symbols are interned for fast comparison
struct sym: val {
	// Although the allocated size of dynamically allocated strings will vary according to the number of characters needed, the
	// declared size of the character array needs to be positive for the statically allocated array of known strings (keywords). It
	// needs to be large enough to accommodate the longest keyword plus null terminator. And the size of the whole structure should
	// be a power of 2 because keyword() needs to divide by that size.
	char z[16 - 1];

	explicit sym(const char* s): val(t_sym) {
		strcpy(z, s);
	}
};

// Keywords are symbols that are known to be important.
enum
{
#define _(x) s_##x,
#include <lo/keywords.h>
	end_s
};

// And statically allocated for fast lookup.
extern sym keywords[];

inline size_t keyword(const val* s) {
	// Assign the difference to an unsigned variable and perform the division explicitly, because ptrdiff_t is a signed type, but
	// unsigned division is slightly faster.
	size_t i = (char*)s - (char*)keywords;
	return i / sizeof(sym);
}

sym* intern(const char* s, size_t n);

inline sym* intern(const char* s) {
	return intern(s, strlen(s));
}

void print(sym* a);
