#include "sort.h"

__declspec(dllimport) struct Str* str(char* s);
__declspec(dllimport) void push(struct Vec* v, void* a);

__declspec(dllexport) void readFile(char* file, struct Vec* v) {
	FILE* f = fopen(file, "r");
	if (!f) {
		perror(file);
		exit(1);
	}
	char buf[100];
	int i = 0;
	int c;
	while ((c = fgetc(f)) != EOF) switch (c) {
		case '\r':
			break;
		case '\n':
			buf[i] = 0;
			push(v, str(buf));
			i = 0;
			break;
		default:
			buf[i++] = c;
			if (i == sizeof buf) {
				fprintf(stderr, "Line too long\n");
				exit(1);
			}
		}
}
