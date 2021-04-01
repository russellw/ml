#include "main.h"

void println(si a) {
  print(a);
  putchar('\n');
}

void print(si a) {
  switch (tag(a)) {
  case t_float:
    printf("%f", floatp(a)->val);
    return;
  case t_int:
    mpz_out_str(stdout, 10, intp(a)->val);
    return;
  case t_rat:
    mpq_out_str(stdout, 10, ratp(a)->val);
    return;
  case t_sym:
    fwrite(symp(a)->v, symp(a)->n, 1, stdout);
    return;
  case t_cons: {
    putchar('(');
    int more = 0;
    while (a != empty) {
      if (more)
        putchar(' ');
      more = 1;
      print(hd(a));
      a = tl(a);
    }
    putchar(')');
    return;
  }
  }
  unreachable;
}
