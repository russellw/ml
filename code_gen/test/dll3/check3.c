int foo();

__declspec(dllexport) int check3() {
	return foo() == 3;
}
