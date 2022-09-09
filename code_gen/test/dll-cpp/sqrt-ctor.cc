#include <math.h>
#include <string.h>

struct LookupTable {
	char header[128];
	double v[100];

	LookupTable() {
		strcpy(header, "this is a lookup table");
		for (int i = 0; i < 100; ++i) v[i] = sqrt(i);
	}
};

LookupTable table;

__declspec(dllexport) double sqrt_ctor(double x) {
	return table.v[(int)x];
}
