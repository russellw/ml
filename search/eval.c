#include "main.h"

si add(si a, si b) {
  switch (tag(a)) {
  case t_float:
    switch (tag(b)) {
    case t_float:
      return ifloat(floatp(a)->val + floatp(b)->val);
    case t_int:
      return ifloat(floatp(a)->val + mpz_get_d(intp(b)->val));
    case t_rat:
      return ifloat(floatp(a)->val + mpq_get_d(ratp(b)->val));
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
      return iint(&r);
    }
    case t_rat: {
      mpq_t a1;
      mpq_init(a1);
      mpq_set_z(a1, intp(a)->val);
      Rat r;
      mpq_init(r.val);
      mpq_add(r.val, a1, ratp(b)->val);
      mpq_clear(a1);
      return irat(&r);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpq_get_d(ratp(a)->val) + floatp(b)->val);
    case t_int: {
      mpq_t b1;
      mpq_init(b1);
      mpq_set_z(b1, intp(b)->val);
      Rat r;
      mpq_init(r.val);
      mpq_add(r.val, ratp(a)->val, b1);
      mpq_clear(b1);
      return irat(&r);
    }
    case t_rat: {
      Rat r;
      mpq_init(r.val);
      mpq_add(r.val, ratp(a)->val, ratp(b)->val);
      return irat(&r);
    }
    }
    break;
  }
  return 0;
}

si eval(si a) { return a; }
