#include <stdio.h>
__declspec(dllimport) int cube(int n);

void square(void) {
	puts("***");
	puts("***");
	puts("***");
}

int main() {
	printf("%d\n", cube(3));
	return 0;
}
