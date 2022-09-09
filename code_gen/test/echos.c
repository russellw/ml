#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
	puts(GetCommandLine());
#endif
	for (int i = 0; i < argc; i++) puts(argv[i]);
	return 0;
}
