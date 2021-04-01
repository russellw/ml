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

static si mkrat(char *s) {
  Rat x;
  mpq_init(x.val);
  mpq_set_str(x.val, s, 10);
  Rat *p = intern_rat(&x);
  mpq_get_str(buf, 10, p->val);
  assert(!strcmp(buf, s));

  Rat y;
  mpq_init(y.val);
  mpq_set_str(y.val, s, 10);
  assert(intern_rat(&y) == p);

  si r = term(p, t_rat);
  assert(tag(r) == t_rat);
  assert(ratp(r) == p);
  return r;
}

void test(void) {
  assert(internz("abc") == internz("abc"));

  si zero = mkint(0);
  si one = mkint(1);
  si two = mkint(2);
  si three = mkint(3);

  si half = mkrat("1/2");
  si third = mkrat("1/3");
  si sixth = mkrat("1/6");
}
#endif
