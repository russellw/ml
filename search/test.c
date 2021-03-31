#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

#ifdef DEBUG
static si mkint(si val) {
  Int x;
  mpz_init_set_si(x.val, val);
  Int *p = intern_int(&x);
  assert(mpz_get_si(p->val) == val);
  Int y;
  mpz_init_set_si(y.val, val);
  assert(intern_int(&y) == p);
  si r = term(p, t_int);
  assert(tag(r) == t_int);
  assert(intp(r) == p);
  return r;
}

void test(void) {
  assert(internz("abc") == internz("abc"));
  si zero = mkint(0);
  si one = mkint(1);
  si two = mkint(2);
  si three = mkint(3);
}
#endif
