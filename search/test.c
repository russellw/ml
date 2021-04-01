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
  // symbols
  assert(internz("abc") == internz("abc"));
  assert(internz("") == internz(""));
  assert(internz("\t") == internz("\t"));
  assert(keyword(internz("round")) == k_round);

  // rationals
  assert(mkrat("1/2") == mkrat("2/4"));
  assert(mkrat("0/1") == mkrat("0/2"));
  assert(mkrat("0/1") == mkrat("-0/2"));
  assert(mkrat("0/1") == mkint(0));
  assert(mkrat("10/1") == mkint(10));

  // arithmetic
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
    div2(mkint(1), mkint(0));
  else
    caught = 1;
  assert(caught);

  caught = 0;
  if (!setjmp(jmpbuf))
    div2(mkrat("2/3"), mkint(0));
  else
    caught = 1;
  assert(caught);

  // floor division
  assert(div_f(mkint(5), mkint(3)) == mkint(1));
  assert(div_f(mkint(-5), mkint(3)) == mkint(-2));
  assert(div_f(mkint(5), mkint(-3)) == mkint(-2));
  assert(div_f(mkint(-5), mkint(-3)) == mkint(1));

  assert(div_f(mkrat("5/997"), mkrat("3/997")) == mkint(1));
  assert(div_f(mkrat("-5/997"), mkrat("3/997")) == mkint(-2));
  assert(div_f(mkrat("5/997"), mkrat("-3/997")) == mkint(-2));
  assert(div_f(mkrat("-5/997"), mkrat("-3/997")) == mkint(1));

  assert(div_f(mkrat("10"), mkrat("-1/10")) == mkrat("-100"));
  assert(div_f(mkrat("1/10"), mkrat("10")) == mkrat("0"));

  assert(rem_f(mkint(5), mkint(3)) == mkint(2));
  assert(rem_f(mkint(-5), mkint(3)) == mkint(1));
  assert(rem_f(mkint(5), mkint(-3)) == mkint(-1));
  assert(rem_f(mkint(-5), mkint(-3)) == mkint(-2));

  assert(rem_f(mkrat("5/997"), mkrat("3/997")) == mkint(2 * 997));
  assert(rem_f(mkrat("-5/997"), mkrat("3/997")) == mkint(1 * 997));
  assert(rem_f(mkrat("5/997"), mkrat("-3/997")) == mkint(-1 * 997));
  assert(rem_f(mkrat("-5/997"), mkrat("-3/997")) == mkint(-2 * 997));

  caught = 0;
  if (!setjmp(jmpbuf))
    div_f(mkrat("2/3"), mkint(0));
  else
    caught = 1;
  assert(caught);

  caught = 0;
  if (!setjmp(jmpbuf))
    div_f(mkrat("2/3"), mkfloat(1));
  else
    caught = 1;
  assert(caught);

  // ceiling division
  assert(div_c(mkint(5), mkint(3)) == mkint(2));
  assert(div_c(mkint(-5), mkint(3)) == mkint(-1));
  assert(div_c(mkint(5), mkint(-3)) == mkint(-1));
  assert(div_c(mkint(-5), mkint(-3)) == mkint(2));

  assert(div_c(mkrat("5/997"), mkrat("3/997")) == mkint(2));
  assert(div_c(mkrat("-5/997"), mkrat("3/997")) == mkint(-1));
  assert(div_c(mkrat("5/997"), mkrat("-3/997")) == mkint(-1));
  assert(div_c(mkrat("-5/997"), mkrat("-3/997")) == mkint(2));

  assert(div_c(mkrat("10"), mkrat("-1/10")) == mkrat("-100"));
  assert(div_c(mkrat("1/10"), mkrat("10")) == mkrat("1"));

  assert(rem_c(mkint(5), mkint(3)) == mkint(-1));
  assert(rem_c(mkint(-5), mkint(3)) == mkint(-2));
  assert(rem_c(mkint(5), mkint(-3)) == mkint(2));
  assert(rem_c(mkint(-5), mkint(-3)) == mkint(1));

  assert(rem_c(mkrat("5/997"), mkrat("3/997")) == mkint(-1 * 997));
  assert(rem_c(mkrat("-5/997"), mkrat("3/997")) == mkint(-2 * 997));
  assert(rem_c(mkrat("5/997"), mkrat("-3/997")) == mkint(2 * 997));
  assert(rem_c(mkrat("-5/997"), mkrat("-3/997")) == mkint(1 * 997));

  // truncating division
  assert(div_t2(mkint(5), mkint(3)) == mkint(5 / 3));
  assert(div_t2(mkint(-5), mkint(3)) == mkint(-5 / 3));
  assert(div_t2(mkint(5), mkint(-3)) == mkint(5 / -3));
  assert(div_t2(mkint(-5), mkint(-3)) == mkint(-5 / -3));
  assert(div_t2(mkint(5), mkint(3)) == mkint(1));
  assert(div_t2(mkint(-5), mkint(3)) == mkint(-1));
  assert(div_t2(mkint(5), mkint(-3)) == mkint(-1));
  assert(div_t2(mkint(-5), mkint(-3)) == mkint(1));

  assert(div_t2(mkrat("5/997"), mkrat("3/997")) == mkint(1));
  assert(div_t2(mkrat("-5/997"), mkrat("3/997")) == mkint(-1));
  assert(div_t2(mkrat("5/997"), mkrat("-3/997")) == mkint(-1));
  assert(div_t2(mkrat("-5/997"), mkrat("-3/997")) == mkint(1));

  assert(div_t2(mkrat("10"), mkrat("-1/10")) == mkrat("-100"));
  assert(div_t2(mkrat("1/10"), mkrat("10")) == mkrat("0"));

  assert(rem_t(mkint(5), mkint(3)) == mkint(5 % 3));
  assert(rem_t(mkint(-5), mkint(3)) == mkint(-5 % 3));
  assert(rem_t(mkint(5), mkint(-3)) == mkint(5 % -3));
  assert(rem_t(mkint(-5), mkint(-3)) == mkint(-5 % -3));
  assert(rem_t(mkint(5), mkint(3)) == mkint(2));
  assert(rem_t(mkint(-5), mkint(3)) == mkint(-2));
  assert(rem_t(mkint(5), mkint(-3)) == mkint(2));
  assert(rem_t(mkint(-5), mkint(-3)) == mkint(-2));

  assert(rem_t(mkrat("5/997"), mkrat("3/997")) == mkint(2 * 997));
  assert(rem_t(mkrat("-5/997"), mkrat("3/997")) == mkint(-2 * 997));
  assert(rem_t(mkrat("5/997"), mkrat("-3/997")) == mkint(2 * 997));
  assert(rem_t(mkrat("-5/997"), mkrat("-3/997")) == mkint(-2 * 997));

  // euclidean division
  assert(div_e(mkint(7), mkint(3)) == mkint(2));
  assert(div_e(mkint(-7), mkint(3)) == mkint(-3));
  assert(div_e(mkint(7), mkint(-3)) == mkint(-2));
  assert(div_e(mkint(-7), mkint(-3)) == mkint(3));

  assert(div_e(mkrat("7/997"), mkrat("3/997")) == mkint(2));
  assert(div_e(mkrat("-7/997"), mkrat("3/997")) == mkint(-3));
  assert(div_e(mkrat("7/997"), mkrat("-3/997")) == mkint(-2));
  assert(div_e(mkrat("-7/997"), mkrat("-3/997")) == mkint(3));

  assert(div_e(mkrat("10"), mkrat("-1/10")) == mkrat("-100"));
  assert(div_e(mkrat("1/10"), mkrat("10")) == mkrat("0"));

  assert(rem_e(mkint(7), mkint(3)) == mkint(1));
  assert(rem_e(mkint(-7), mkint(3)) == mkint(2));
  assert(rem_e(mkint(7), mkint(-3)) == mkint(1));
  assert(rem_e(mkint(-7), mkint(-3)) == mkint(2));

  assert(rem_e(mkrat("7/997"), mkrat("3/997")) == mkint(1 * 997));
  assert(rem_e(mkrat("-7/997"), mkrat("3/997")) == mkint(2 * 997));
  assert(rem_e(mkrat("7/997"), mkrat("-3/997")) == mkint(1 * 997));
  assert(rem_e(mkrat("-7/997"), mkrat("-3/997")) == mkint(2 * 997));

  // unary minus
  assert(minus(mkfloat(2.5)) == mkfloat(-2.5));
  assert(minus(mkint(2)) == mkint(-2));
  assert(minus(mkrat("-1/7")) == mkrat("1/7"));

  // ceiling
  assert(ceil1(mkrat("0")) == mkrat("0"));
  assert(ceil1(mkrat("1/10")) == mkrat("1"));
  assert(ceil1(mkrat("5/10")) == mkrat("1"));
  assert(ceil1(mkrat("9/10")) == mkrat("1"));
  assert(ceil1(mkrat("-1/10")) == mkrat("0"));
  assert(ceil1(mkrat("-5/10")) == mkrat("0"));
  assert(ceil1(mkrat("-9/10")) == mkrat("0"));

  assert(ceil1(mkfloat(0)) == mkfloat(0));
  assert(ceil1(mkfloat(0.1)) == mkfloat(1));
  assert(ceil1(mkfloat(0.5)) == mkfloat(1));
  assert(ceil1(mkfloat(0.9)) == mkfloat(1));
  assert(ceil1(mkfloat(-0.1)) == mkfloat(-0.0));
  assert(ceil1(mkfloat(-0.5)) == mkfloat(-0.0));
  assert(ceil1(mkfloat(-0.9)) == mkfloat(-0.0));

  // floor
  assert(floor1(mkrat("0")) == mkrat("0"));
  assert(floor1(mkrat("1/10")) == mkrat("0"));
  assert(floor1(mkrat("5/10")) == mkrat("0"));
  assert(floor1(mkrat("9/10")) == mkrat("0"));
  assert(floor1(mkrat("-1/10")) == mkrat("-1"));
  assert(floor1(mkrat("-5/10")) == mkrat("-1"));
  assert(floor1(mkrat("-9/10")) == mkrat("-1"));

  assert(floor1(mkfloat(0)) == mkfloat(0));
  assert(floor1(mkfloat(0.1)) == mkfloat(0));
  assert(floor1(mkfloat(0.5)) == mkfloat(0));
  assert(floor1(mkfloat(0.9)) == mkfloat(0));
  assert(floor1(mkfloat(-0.1)) == mkfloat(-1.0));
  assert(floor1(mkfloat(-0.5)) == mkfloat(-1.0));
  assert(floor1(mkfloat(-0.9)) == mkfloat(-1.0));

  // truncate
  assert(trunc1(mkrat("0")) == mkrat("0"));
  assert(trunc1(mkrat("1/10")) == mkrat("0"));
  assert(trunc1(mkrat("5/10")) == mkrat("0"));
  assert(trunc1(mkrat("9/10")) == mkrat("0"));
  assert(trunc1(mkrat("-1/10")) == mkrat("0"));
  assert(trunc1(mkrat("-5/10")) == mkrat("0"));
  assert(trunc1(mkrat("-9/10")) == mkrat("0"));

  assert(trunc1(mkfloat(0)) == mkfloat(0));
  assert(trunc1(mkfloat(0.1)) == mkfloat(0));
  assert(trunc1(mkfloat(0.5)) == mkfloat(0));
  assert(trunc1(mkfloat(0.9)) == mkfloat(0));
  assert(trunc1(mkfloat(-0.1)) == mkfloat(-0.0));
  assert(trunc1(mkfloat(-0.5)) == mkfloat(-0.0));
  assert(trunc1(mkfloat(-0.9)) == mkfloat(-0.0));

  // round
  assert(round1(mkrat("0")) == mkrat("0"));
  assert(round1(mkrat("1/10")) == mkrat("0"));
  assert(round1(mkrat("5/10")) == mkrat("0"));
  assert(round1(mkrat("9/10")) == mkrat("1"));
  assert(round1(mkrat("-1/10")) == mkrat("0"));
  assert(round1(mkrat("-5/10")) == mkrat("0"));
  assert(round1(mkrat("-9/10")) == mkrat("-1"));
  assert(round1(mkrat("15/10")) == mkrat("2"));
  assert(round1(mkrat("25/10")) == mkrat("2"));
  assert(round1(mkrat("35/10")) == mkrat("4"));
  assert(round1(mkrat("45/10")) == mkrat("4"));

  assert(round1(mkfloat(0)) == mkfloat(0));
  assert(round1(mkfloat(0.1)) == mkfloat(0));
  assert(round1(mkfloat(0.9)) == mkfloat(1.0));
  assert(round1(mkfloat(-0.1)) == mkfloat(-0.0));
  assert(round1(mkfloat(-0.9)) == mkfloat(-1.0));

  // abs
  assert(abs1(mkrat("0")) == mkrat("0"));
  assert(abs1(mkrat("1/10")) == mkrat("1/10"));
  assert(abs1(mkrat("5/10")) == mkrat("5/10"));
  assert(abs1(mkrat("9/10")) == mkrat("9/10"));
  assert(abs1(mkrat("-1/10")) == mkrat("1/10"));
  assert(abs1(mkrat("-5/10")) == mkrat("5/10"));
  assert(abs1(mkrat("-9/10")) == mkrat("9/10"));

  assert(abs1(mkfloat(0)) == mkfloat(0));
  assert(abs1(mkfloat(0.1)) == mkfloat(0.1));
  assert(abs1(mkfloat(0.5)) == mkfloat(0.5));
  assert(abs1(mkfloat(0.9)) == mkfloat(0.9));
  assert(abs1(mkfloat(-0.1)) == mkfloat(0.1));
  assert(abs1(mkfloat(-0.5)) == mkfloat(0.5));
  assert(abs1(mkfloat(-0.9)) == mkfloat(0.9));

  // lt
  assert(lt(mkint(1), mkint(2)));
  assert(!lt(mkint(1), mkint(1)));
  assert(!lt(mkint(2), mkint(1)));
  assert(!lt(mkint(-1), mkint(-2)));
  assert(!lt(mkint(-1), mkint(-1)));
  assert(lt(mkint(-2), mkint(-1)));

  assert(lt(mkfloat(1), mkfloat(2)));
  assert(!lt(mkfloat(1), mkfloat(1)));
  assert(!lt(mkfloat(2), mkfloat(1)));
  assert(!lt(mkfloat(-1), mkfloat(-2)));
  assert(!lt(mkfloat(-1), mkfloat(-1)));
  assert(lt(mkfloat(-2), mkfloat(-1)));

  assert(lt(mkint(1), mkfloat(2)));
  assert(!lt(mkint(1), mkfloat(1)));
  assert(!lt(mkint(2), mkfloat(1)));
  assert(!lt(mkint(-1), mkfloat(-2)));
  assert(!lt(mkint(-1), mkfloat(-1)));
  assert(lt(mkint(-2), mkfloat(-1)));

  assert(lt(mkfloat(1), mkint(2)));
  assert(!lt(mkfloat(1), mkint(1)));
  assert(!lt(mkfloat(2), mkint(1)));
  assert(!lt(mkfloat(-1), mkint(-2)));
  assert(!lt(mkfloat(-1), mkint(-1)));
  assert(lt(mkfloat(-2), mkint(-1)));

  assert(lt(mkrat("1/997"), mkrat("2/997")));
  assert(!lt(mkrat("1/997"), mkrat("1/997")));
  assert(!lt(mkrat("2/997"), mkrat("1/997")));
  assert(!lt(mkrat("-1/997"), mkrat("-2/997")));
  assert(!lt(mkrat("-1/997"), mkrat("-1/997")));
  assert(lt(mkrat("-2/997"), mkrat("-1/997")));

  assert(lt(mkrat("3/10"), mkrat("10/10")));
  assert(!lt(mkrat("13/10"), mkrat("10/10")));

  // le
  assert(le(mkint(1), mkint(2)));
  assert(le(mkint(1), mkint(1)));
  assert(!le(mkint(2), mkint(1)));
  assert(!le(mkint(-1), mkint(-2)));
  assert(le(mkint(-1), mkint(-1)));
  assert(le(mkint(-2), mkint(-1)));

  assert(le(mkfloat(1), mkfloat(2)));
  assert(le(mkfloat(1), mkfloat(1)));
  assert(!le(mkfloat(2), mkfloat(1)));
  assert(!le(mkfloat(-1), mkfloat(-2)));
  assert(le(mkfloat(-1), mkfloat(-1)));
  assert(le(mkfloat(-2), mkfloat(-1)));

  assert(le(mkint(1), mkfloat(2)));
  assert(le(mkint(1), mkfloat(1)));
  assert(!le(mkint(2), mkfloat(1)));
  assert(!le(mkint(-1), mkfloat(-2)));
  assert(le(mkint(-1), mkfloat(-1)));
  assert(le(mkint(-2), mkfloat(-1)));

  assert(le(mkfloat(1), mkint(2)));
  assert(le(mkfloat(1), mkint(1)));
  assert(!le(mkfloat(2), mkint(1)));
  assert(!le(mkfloat(-1), mkint(-2)));
  assert(le(mkfloat(-1), mkint(-1)));
  assert(le(mkfloat(-2), mkint(-1)));

  assert(le(mkrat("1/997"), mkrat("2/997")));
  assert(le(mkrat("1/997"), mkrat("1/997")));
  assert(!le(mkrat("2/997"), mkrat("1/997")));
  assert(!le(mkrat("-1/997"), mkrat("-2/997")));
  assert(le(mkrat("-1/997"), mkrat("-1/997")));
  assert(le(mkrat("-2/997"), mkrat("-1/997")));

  assert(le(mkrat("3/10"), mkrat("10/10")));
  assert(!le(mkrat("13/10"), mkrat("10/10")));

  // truth
  assert(istrue(mkint(5)));
  assert(!istrue(mkint(0)));
  assert(istrue(mkrat("-1/65535")));
  assert(istrue(mkfloat(0.1)));
  assert(!istrue(mkfloat(0.0)));

  // inexact
  assert(inexact(mkint(-4)) == mkfloat(-4));
  assert(inexact(mkfloat(-4)) == mkfloat(-4));
  assert(inexact(mkrat("1/2")) == mkfloat(.5));

  // exact
  assert(exact(mkint(-4)) == mkint(-4));
  assert(exact(mkfloat(-4)) == mkint(-4));
  assert(exact(mkfloat(.5)) == mkrat("1/2"));
  assert(exact(mkfloat(-0.0)) == mkint(0));

  // cons
  assert(tag(empty) == t_cons);

  caught = 0;
  if (!setjmp(jmpbuf))
    cons(mkrat("2/3"), mkfloat(1));
  else
    caught = 1;
  assert(caught);

  assert(cons(mkint(1), empty) == cons(mkint(1), empty));
  assert(cons(mkint(1), cons(mkint(2), empty)) ==
         cons(mkint(1), cons(mkint(2), empty)));
}
#endif
