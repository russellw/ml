#include <stdio.h>
#include <string.h>

unsigned long long crc32(unsigned long long c, char* s, unsigned n);

int main(int argc, char** argv) {
	if (argc < 2) {
		puts("Usage: call-crc32 string");
		return 1;
	}
	printf("%llu\n", crc32(0, argv[1], strlen(argv[1])));
	return 0;
}
