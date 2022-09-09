#include <math.h>
#include <windows.h>

double table[100];

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
	switch (fdwReason) {
	case DLL_PROCESS_ATTACH:
		for (int i = 0; i < 100; ++i) table[i] = sqrt(i);
		break;
	}
	return TRUE;
}

extern "C" __declspec(dllexport) double sqrt_load(double x) {
	return table[(int)x];
}
