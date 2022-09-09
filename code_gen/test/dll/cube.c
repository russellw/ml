int square(int n);

__declspec(dllexport) int cube(int n) {
	return square(n) * n;
}
