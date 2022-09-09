int foo();

__declspec(dllexport) int check1() {
	return foo() == 1;
}
