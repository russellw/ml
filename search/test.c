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
  assert(add(mkfloat(0.5), mkrat("1/2")) == mkfloat(1.0));
  assert(add(mkint(-1), mkrat("1/2")) == mkrat("-1/2"));
  assert(add(mkrat("1/2"), mkint(-1)) == mkrat("-1/2"));

  si caught = 0;
  if (!setjmp(jmpbuf))
    add(internz("a"), mkint(1));
  else
    caught = 1;
  assert(caught);

  assert(sub(mkfloat(0.5), mkfloat(0.5)) == mkfloat(0.0));
  assert(sub(mkfloat(1.0), mkfloat(2.0)) == mkfloat(-1.0));
  assert(sub(mkfloat(10.0), mkfloat(-0.5)) == mkfloat(10.5));
  assert(sub(mkint(11), mkint(12)) == mkint(-1));
  assert(sub(mkint(11), mkint(-12)) == mkint(23));
  assert(sub(mkint(0), mkint(-12)) == mkint(12));
  assert(sub(mkfloat(1.0), mkint(5)) == mkfloat(-4.0));
  assert(sub(mkint(100), mkfloat(-1.0)) == mkfloat(101.0));
  assert(sub(mkrat("1/3"), mkrat("1/2")) == mkrat("-1/6"));
  assert(sub(mkrat("1/2"), mkfloat(0.5)) == mkfloat(0.0));
  assert(sub(mkfloat(0.5), mkrat("1/2")) == mkfloat(0.0));
  assert(sub(mkint(-1), mkrat("1/2")) == mkrat("-3/2"));
  assert(sub(mkrat("1/2"), mkint(-1)) == mkrat("3/2"));

  assert(mul(mkfloat(0.5), mkfloat(0.5)) == mkfloat(0.25));
  assert(mul(mkfloat(1.0), mkfloat(2.0)) == mkfloat(2.0));
  assert(mul(mkfloat(10.0), mkfloat(-0.5)) == mkfloat(-5.0));
  assert(mul(mkint(11), mkint(12)) == mkint(132));
  assert(mul(mkint(11), mkint(-12)) == mkint(-132));
  assert(mul(mkint(0), mkint(-12)) == mkint(0));
  assert(mul(mkfloat(1.0), mkint(5)) == mkfloat(5));
  assert(mul(mkint(100), mkfloat(-1.0)) == mkfloat(-100.0));
  assert(mul(mkrat("1/3"), mkrat("1/2")) == mkrat("1/6"));
  assert(mul(mkrat("1/2"), mkfloat(0.5)) == mkfloat(0.25));
  assert(mul(mkfloat(0.5), mkrat("1/2")) == mkfloat(0.25));
  assert(mul(mkint(-1), mkrat("1/2")) == mkrat("-1/2"));
  assert(mul(mkrat("1/2"), mkint(-1)) == mkrat("-1/2"));

  assert(div2(mkfloat(0.5), mkfloat(0.5)) == mkfloat(1));
  assert(div2(mkfloat(1.0), mkfloat(2.0)) == mkfloat(.5));
  assert(div2(mkfloat(10.0), mkfloat(-0.5)) == mkfloat(-20.0));
  assert(div2(mkint(11), mkint(12)) == mkrat("11/12"));
  assert(div2(mkint(11), mkint(-12)) == mkrat("-11/12"));
  assert(div2(mkint(0), mkint(-12)) == mkint(0));
  assert(div2(mkint(10), mkint(5)) == mkint(2));
  assert(div2(mkint(100), mkfloat(-1.0)) == mkfloat(-100.0));
  assert(div2(mkrat("1/3"), mkrat("1/2")) == mkrat("2/3"));
  assert(div2(mkrat("1/2"), mkfloat(0.5)) == mkfloat(1));
  assert(div2(mkfloat(0.5), mkrat("1/2")) == mkfloat(1));
  assert(div2(mkint(-1), mkrat("1/2")) == mkrat("-2"));
  assert(div2(mkrat("1/2"), mkint(-1)) == mkrat("-1/2"));

  caught = 0;
  if (!setjmp(jmpbuf))
    div2( mkint(1),mkint(0));
  else
    caught = 1;
  assert(caught);

  caught = 0;
  if (!setjmp(jmpbuf))
    div2( mkrat("2/3"),mkint(0));
  else
    caught = 1;
  assert(caught);
}
#endif
