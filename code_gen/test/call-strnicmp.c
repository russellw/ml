#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
	if (argc == 2 && !strnicmp(argv[1], "foo", 3)) puts("you said foo");
	else
		puts("you did not say foo");
	return 0;
}
