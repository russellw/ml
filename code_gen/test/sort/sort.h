#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Str {
	int n;
	char v[];
};

struct Vec {
	int cap, n;
	void** p;
};
