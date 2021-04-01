#include "main.h"

si add(si a, si b) {
  switch (tag(a)) {
  case t_float:
    switch (tag(b)) {
    case t_float:
      return ifloat(floatp(a)->val + floatp(b)->val);
    case t_int:
      return ifloat(floatp(a)->val + mpz_get_d(intp(b)->val));
    }
    break;
  case t_int:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpz_get_d(intp(a)->val) + floatp(b)->val);
    case t_int: {
      Int r;
      mpz_init(r.val);
      mpz_add(r.val, intp(a)->val, intp(b)->val);
      return term(iint(&r), t_int);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpq_get_d(ratp(a)->val) + floatp(b)->val);
    case t_rat: {
      Rat r;
      mpq_init(r.val);
      mpq_add(r.val, ratp(a)->val, ratp(b)->val);
      return term(irat(&r), t_rat);
    }
    }
    break;
  }
  return 0;
}

si eval(si a) { return a; }
