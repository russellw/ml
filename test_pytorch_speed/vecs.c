// results on a 3 GHz CPU
// counting only FP multiplies/sec

// with just /O2

// vector length 10
// 2083e6

// vector length 100
// 1219e6

// vector length 1000
// 917e6

// vector length 10000
// 909e6

// vector length 100000
// 901e6

// vector length 1000000
// 885e6

// vector length 10000000
// 862e6

// vector length 100000000
// 671e6

// with /O2 /arch:AVX2

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t parse(char *s) {
  errno = 0;
  size_t n = strtoull(s, 0, 10);
  if (errno) {
    perror(s);
    exit(1);
  }
  return n;
}

void *xmalloc(size_t bytes) {
  void *p = malloc(bytes);
  if (!p) {
    perror("malloc");
    exit(1);
  }
  return p;
}

float frand(void) { return rand() / (RAND_MAX + 1.0); }

float *rand_vec(size_t n) {
  float *a = malloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++)
    a[i] = frand();
  return a;
}

float mul(float *a, float *b, size_t n) {
  float r = 0;
  for (size_t i = 0; i < n; i++)
    r += a[i] * b[i];
  return r;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: vecs <size> <times>\n");
    return 1;
  }
  size_t size = parse(argv[1]);
  size_t i = parse(argv[2]);
  float *a = rand_vec(size);
  float *b = rand_vec(size);

  float x = 0;
  while (i--) {
    x += mul(a, b, size);
  }
  printf("%f\n", x);
  return 0;
}
