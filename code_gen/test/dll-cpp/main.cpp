#include <math.h>
#include <stdio.h>
#include <windows.h>

__declspec(dllimport) double sqrt_plain(double x);
__declspec(dllimport) double sqrt_ctor(double x);
__declspec(dllimport) double sqrt_dllmain(double x);

int main() {
	printf("%f\n", sqrt(50.0));
	printf("%f\n", sqrt_plain(50.0));
	printf("%f\n", sqrt_ctor(50.0));
	printf("%f\n", sqrt_dllmain(50.0));
	auto m = LoadLibrary("sqrt-load.dll");
	printf("%p\n", m);
	typedef double(__cdecl * F)(double);
	auto f = (F)GetProcAddress(m, "sqrt_load");
	printf("%p\n", f);
	printf("%f\n", f(50.0));
	return 0;
}
