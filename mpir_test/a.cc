#include <stdio.h>
#include <gmp.h>

int main(int argc, const char **argv) {
  mpz_t x;
  mpz_init_set_str(x,"123",10);
  mpz_out_str(stdout,10,x);
  putchar('\n');
  return 0;
}
