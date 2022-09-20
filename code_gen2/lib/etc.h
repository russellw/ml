const size_t bufsz = 0x1000;
extern char buf[];

// SORT
[[noreturn]] void err(const char* file, const char* s, const char* t, const char* msg);
size_t fnv(const void* p, size_t bytes);
void readFile(const char* file, vector<char>& text);
void* xcalloc(size_t n, size_t size);
void* xmalloc(size_t bytes);
///

inline bool isDigit(int c) { return '0' <= c && c <= '9'; }
inline bool isLower(int c) { return 'a' <= c && c <= 'z'; }
inline bool isUpper(int c) { return 'A' <= c && c <= 'Z'; }
inline bool isAlpha(int c) { return isLower(c) || isUpper(c); }
inline bool isAlnum(int c) { return isAlpha(c) || isDigit(c); }
inline bool isId(int c) { return isAlnum(c) || c == '_'; }

constexpr bool isPow2(size_t n) {
	assert(n);
	return !(n & (n - 1));
}
