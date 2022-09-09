#include <stdio.h>

int check1();
int check2();
int check3();

int main() {
	//this does not work; static linking only picks up one version of foo across all static libraries.
	//This confirms that, unlike dynamic libraries, static libraries do not require
	//two linking phases
	if (!check1()) {
		puts("error");
		return 1;
	}
	if (!check2()) {
		puts("error");
		return 1;
	}
	if (!check3()) {
		puts("error");
		return 1;
	}
	puts("ok");
	return 0;
}
