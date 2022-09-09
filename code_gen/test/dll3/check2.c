int foo();

__declspec(dllexport) int check2() {
	return foo() == 2;
}
