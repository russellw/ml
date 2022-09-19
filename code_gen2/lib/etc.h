const size_t bufsz = 0x1000;
extern char buf[];

// SORT
size_t fnv(const void* p, size_t bytes);
void readFile(const char* file, vector<char>& text);
void* xcalloc(size_t n, size_t size);
void* xmalloc(size_t bytes);
///

constexpr bool isPow2(size_t n) {
	assert(n);
	return !(n & (n - 1));
}
