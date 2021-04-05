#include "main.h"

#ifdef DEBUG
static si mkfloat(double val) {
  si r = ifloat(val);
  assert(floatp(r)->val == val);
  return r;
}

static si mkrat(char *s) {
  Rat x;
  mpq_init(x.val);
  mpq_set_str(x.val, s, 0);
  return irat(&x);
}

static si parse1(void) {
  vec v;
  vinit(&v);
  parse(&v);
  return list(&v);
}

void test(void) {
  // symbols
  assert(internz("abc") == internz("abc"));
  assert(internz("") == internz(""));
  assert(internz("\t") == internz("\t"));
  assert(keyword(internz("round")) == w_round);

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
  assert(tag(nil) == t_cons);

  assert(cons(mkint(1), nil) == cons(mkint(1), nil));
  assert(cons(mkint(1), cons(mkint(2), nil)) ==
         cons(mkint(1), cons(mkint(2), nil)));
  assert(list4(internz("if"), mkint(1), mkint(2), mkint(3)) ==
         list4(internz("if"), mkint(1), mkint(2), mkint(3)));

  // lists
  assert(hd(cons(mkint(1), nil)) == mkint(1));
  assert(tl(cons(mkint(1), nil)) == nil);

  // constant
  assert(eval(nil, mkint(5)) == mkint(5));

  // if
  assert(eval(nil, list4(internz("if"), mkint(1), mkint(2), mkint(3))) ==
         mkint(2));
  assert(eval(nil, list4(internz("if"), mkint(0), mkint(2), mkint(3))) ==
         mkint(3));

  // quote
  assert(eval(nil, list2(internz("quote"), internz("foo"))) == internz("foo"));

  // not
  assert(eval(nil, list2(internz("not"), mkint(2))) == mkint(0));
  assert(eval(nil, list2(internz("not"), mkint(0))) == mkint(1));

  // and
  assert(eval(nil, list1(internz("and"))) == mkint(1));
  assert(eval(nil, list2(internz("and"), mkint(0))) == mkint(0));
  assert(eval(nil, list2(internz("and"), mkint(1))) == mkint(1));
  assert(eval(nil, list3(internz("and"), mkint(0), mkint(0))) == mkint(0));
  assert(eval(nil, list3(internz("and"), mkint(0), mkint(1))) == mkint(0));
  assert(eval(nil, list3(internz("and"), mkint(1), mkint(0))) == mkint(0));
  assert(eval(nil, list3(internz("and"), mkint(1), mkint(1))) == mkint(1));

  // or
  assert(eval(nil, list1(internz("or"))) == mkint(0));
  assert(eval(nil, list2(internz("or"), mkint(0))) == mkint(0));
  assert(eval(nil, list2(internz("or"), mkint(1))) == mkint(1));
  assert(eval(nil, list3(internz("or"), mkint(0), mkint(0))) == mkint(0));
  assert(eval(nil, list3(internz("or"), mkint(0), mkint(1))) == mkint(1));
  assert(eval(nil, list3(internz("or"), mkint(1), mkint(0))) == mkint(1));
  assert(eval(nil, list3(internz("or"), mkint(1), mkint(1))) == mkint(1));

  // parsing
  assert(mkrat("0b10000") == mkint(16));
  assert(mkrat(" -0B10000 ") == mkint(-16));
  assert(mkrat("0x100") == mkrat("256"));
  assert(mkrat("0x1/0x100") == mkrat("1/256"));

  // vec
  vec v;
  vinit(&v);
  assert(v.n == 0);
  assert(v.n <= v.cap);
  for (int i = 0; i < 100; i++) {
    vpush(&v, i * 10);
    assert(v.n == i + 1);
    assert(v.n <= v.cap);
    assert(v.p[0] == 0);
    assert(v.p[v.n - 1] == i * 10);
  }
  vfree(&v);
  assert(_CrtCheckMemory());

  // parsing
  txt = "123";
  assert(parse1() == list1(mkint(123)));

  txt = "1 (2 3 4) 5";
  assert(parse1() ==
         list3(mkint(1), list3(mkint(2), mkint(3), mkint(4)), mkint(5)));

  txt = "1 [2 3 4] 5";
  assert(parse1() ==
         list3(mkint(1), list3(mkint(2), mkint(3), mkint(4)), mkint(5)));

  txt = "'A'";
  assert(parse1() == list1(mkint(65)));

  txt = "\"ABC\"";
  assert(parse1() == list1(list2(internz("quote"),
                                 list3(mkint(65), mkint(66), mkint(67)))));

  txt = "123/456";
  assert(parse1() == list1(mkrat("123/456")));

  txt = "-123/456";
  assert(parse1() == list1(mkrat("-123/456")));

  txt = "0x100";
  assert(parse1() == list1(mkrat("256")));

  txt = "0x100/100";
  assert(parse1() == list1(mkrat("256/100")));

  txt = "0x100/0x100";
  assert(parse1() == list1(mkrat("256/256")));

  txt = "0b100";
  assert(parse1() == list1(mkrat("4")));

  txt = "0b100/100";
  assert(parse1() == list1(mkrat("4/100")));

  txt = "0b100/0b100";
  assert(parse1() == list1(mkrat("4/4")));

  txt = ".5";
  assert(parse1() == list1(mkfloat(.5)));

  txt = "0.5";
  assert(parse1() == list1(mkfloat(.5)));

  txt = "0.5e0";
  assert(parse1() == list1(mkfloat(.5)));

  txt = "0.0005e3";
  assert(parse1() == list1(mkfloat(.5)));

  txt = "0.0005E+3";
  assert(parse1() == list1(mkfloat(.5)));

  txt = "0.0005e-3";
  assert(parse1() == list1(mkfloat(.0000005)));

  txt = "1.0 2.0";
  assert(parse1() == list2(mkfloat(1), mkfloat(2)));

  txt = "0x100.1p5";
  assert(parse1() == list1(mkfloat(8194)));

  txt = "a";
  assert(parse1() == list1(internz("a")));

  // environments
  si frame = list3(list2(internz("a"), mkint(1)), list2(internz("b"), mkint(2)),
                   list2(internz("c"), mkint(3)));
  assert(frame == zip(list3(internz("a"), internz("b"), internz("c")),
                      list4(mkint(1), mkint(2), mkint(3), mkint(4))));
  si env = list1(frame);
  assert(get(env, internz("a")) == list2(internz("a"), mkint(1)));
  assert(get(env, internz("b")) == list2(internz("b"), mkint(2)));
  assert(get(env, internz("c")) == list2(internz("c"), mkint(3)));

  frame = list1(list2(internz("a"), mkint(4)));
  env = cons(frame, env);
  assert(get(env, internz("a")) == list2(internz("a"), mkint(4)));
  assert(get(env, internz("b")) == list2(internz("b"), mkint(2)));
  assert(get(env, internz("c")) == list2(internz("c"), mkint(3)));

  assert(eval(env, internz("a")) == mkint(4));

  si f = list3(env, list1(internz("x")),
               list4(internz("if"), internz("x"), mkint(8), mkint(9)));
  assert(apply(f, list1(mkint(1))) == mkint(8));
  assert(apply(f, list1(mkint(0))) == mkint(9));

  assert(eval(env, list3(internz("+"), mkint(100), mkint(200))) == mkint(300));
  assert(eval(env, list2(internz("minus"), mkint(100))) == mkint(-100));

  f = list3(internz("\\"), list1(internz("x")),
            list2(internz("minus"), internz("x")));
  assert(eval(env, list2(f, mkint(100))) == mkint(-100));

  f = list3(internz("\\"), internz("x"), list2(internz("minus"), internz("x")));
  assert(eval(env, list2(f, mkint(100))) == mkint(-100));
}
#endif
