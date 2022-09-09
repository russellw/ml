#include <stdio.h>

int check1();
int check2();
int check3();

int main() {
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
