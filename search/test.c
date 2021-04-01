#include "main.h"

#ifdef DEBUG
static si mkfloat(double val) {
  si r = ifloat(val);
  assert(floatp(r)->val == val);
  return r;
}

static si mkint(si val) {
  Int x;
  mpz_init_set_si(x.val, val);
  si r = iint(&x);
  assert(mpz_get_si(intp(r)->val) == val);
  return r;
}

static si mkrat(char *s) {
  Rat x;
  mpq_init(x.val);
  mpq_set_str(x.val, s, 10);
  return irat(&x);
}

void test(void) {
  assert(internz("abc") == internz("abc"));
  assert(internz("") == internz(""));
  assert(internz("\t") == internz("\t"));

  assert(mkrat("1/2") == mkrat("2/4"));
  assert(mkrat("0/1") == mkrat("0/2"));
  assert(mkrat("0/1") == mkrat("-0/2"));
  assert(mkrat("0/1") == mkint(0));
  assert(mkrat("10/1") == mkint(10));

  assert(add(mkfloat(0.5), mkfloat(0.5)) == mkfloat(1.0));
  assert(add(mkfloat(1.0), mkfloat(2.0)) == mkfloat(3.0));
  assert(add(mkfloat(10.0), mkfloat(-0.5)) == mkfloat(9.5));
  assert(add(mkint(11), mkint(12)) == mkint(23));
  assert(add(mkint(11), mkint(-12)) == mkint(-1));
  assert(add(mkint(0), mkint(-12)) == mkint(-12));
  assert(add(mkfloat(1.0), mkint(5)) == mkfloat(6.0));
  assert(add(mkint(100), mkfloat(-1.0)) == mkfloat(99.0));
  assert(add(mkrat("1/3"), mkrat("1/2")) == mkrat("5/6"));
  assert(add(mkrat("1/2"), mkfloat(0.5)) == mkfloat(1.0));
}
#endif
