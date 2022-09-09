#include "all.h"

int main(int argc, char** argv) {
	world[WORLD_SIZE / 2] = 1;
	for (int i = 0; i < 60; i++) {
		for (int j = 0; j < WORLD_SIZE; j++) {
			if (j) putchar(' ');
			putchar(world[j] ? '*' : '.');
		}
		putchar('\n');
		update(rule110);
	}
	return 0;
}
