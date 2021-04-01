#include "main.h"

#ifdef DEBUG
static si mkfloat(double val) {
  Float *p = intern_float(val);
  assert(p->val == val);

  assert(intern_float(val) == p);

  si r = term(p, t_float);
  assert(tag(r) == t_float);
  assert(floatp(r) == p);
  return r;
}

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

  assert(add(mkfloat(0.5), mkfloat(0.5)) == mkfloat(1.0));
  assert(add(mkfloat(1.0), mkfloat(2.0)) == mkfloat(3.0));
  assert(add(mkfloat(10.0), mkfloat(-0.5)) == mkfloat(9.5));

  assert(add(mkint(11), mkint(12)) == mkint(23));
  assert(add(mkint(11), mkint(-12)) == mkint(-1));
  assert(add(mkint(0), mkint(-12)) == mkint(-12));
}
#endif
