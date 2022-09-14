void* xmalloc(size_t bytes);
void* xcalloc(size_t n, size_t size);
size_t fnv(const void* p, size_t bytes);

constexpr bool isPow2(size_t n) {
	assert(n);
	return !(n & n - 1);
}
