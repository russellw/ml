// results on a 3 GHz CPU
// counting only FP multiplies/sec

// with just /O2


// with /O2 /arch:AVX2


#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 16

void *xmalloc(size_t bytes) {
  void *p = malloc(bytes);
  if (!p) {
    perror("malloc");
    exit(1);
  }
  return p;
}

float *mk_vec(void) {
  float *a = malloc(N * sizeof(float));
  for (size_t i = 0; i < N; i++)
    a[i] = 1.0;
  return a;
}

float mul(float *a, float *b) {
  float r = 0;
  for (size_t i = 0; i < N; i++)
    r += a[i] * b[i];
  return r;
}

int main(int argc, char **argv) {
  size_t i = 100000000ull;
  float *a = mk_vec();
  float *b = mk_vec();

  float x = 0;
  while (i--) {
    x += mul(a, b);
  }
  printf("%f\n", x);
  return 0;
}
