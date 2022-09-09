#include "all.h"

static int get(char* w, int i) {
	if (i < 0) i += WORLD_SIZE;
	return w[i % WORLD_SIZE];
}

void update(char* rule) {
	char world1[WORLD_SIZE];
	for (int i = 0; i < WORLD_SIZE; i++) {
		int j = get(world, i - 1) << 2 | get(world, i) << 1 | get(world, i + 1);
		world1[i] = rule[j];
	}
	memcpy(world, world1, sizeof world1);
}
