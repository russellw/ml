#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

namespace {
bool isConst(term a) {
  switch (tag(a)) {
  case term::Int:
  case term::Rat:
  case term::Real:
    return 1;
  }
  return 0;
}
} // namespace

term simplify(term a) {
  ck(a);
  if (!isCompound(a))
    return a;
  auto n = size(a);
  vec<term> v(n);
  for (si i = 0; i != n; ++i)
    v[i] = simplify(at(a, i));
  a = intern(tag(a), v);
  auto x = v[0];
  term y;
  if (n > 1)
    y = v[1];
  switch (tag(a)) {
  case term::Add:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int: {
      Int r;
      mpz_init(r.val);
      mpz_add(r.val, ((Int *)rest(x))->val, ((Int *)rest(y))->val);
      return tag(term::Int, intern(r));
    }
    case term::Rat:
    case term::Real: {
      Rat r;
      mpq_init(r.val);
      mpq_add(r.val, ((Rat *)rest(x))->val, ((Rat *)rest(y))->val);
      return tag(tag(x), intern(r));
    }
    }
    break;
  case term::Ceil:
    switch (tag(x)) {
    case term::Int:
      return x;
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      Rat q;
      mpq_init(q.val);
      mpz_cdiv_q(mpq_numref(q.val), mpq_numref(x1->val), mpq_denref(x1->val));
      return tag(tag(x), intern(q));
    }
    }
    break;
  case term::Div:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Rat:
    case term::Real: {
      Rat r;
      mpq_init(r.val);
      mpq_div(r.val, ((Rat *)rest(x))->val, ((Rat *)rest(y))->val);
      return tag(tag(x), intern(r));
    }
    }
    break;
  case term::DivE:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int: {
      Int q;
      mpz_init(q.val);
      mpz_ediv_q(q.val, ((Int *)rest(x))->val, ((Int *)rest(y))->val);
      return tag(term::Int, intern(q));
    }
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      auto y1 = (Rat *)rest(y);

      mpz_t xnum_yden;
      mpz_init(xnum_yden);
      mpz_mul(xnum_yden, mpq_numref(x1->val), mpq_denref(y1->val));

      mpz_t xden_ynum;
      mpz_init(xden_ynum);
      mpz_mul(xden_ynum, mpq_denref(x1->val), mpq_numref(y1->val));

      Rat q;
      mpq_init(q.val);
      mpz_ediv_q(mpq_numref(q.val), xnum_yden, xden_ynum);
      a = tag(tag(x), intern(q));

      mpz_clear(xden_ynum);
      mpz_clear(xnum_yden);
      return a;
    }
    }
    break;
  case term::DivF:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int: {
      Int q;
      mpz_init(q.val);
      mpz_fdiv_q(q.val, ((Int *)rest(x))->val, ((Int *)rest(y))->val);
      return tag(term::Int, intern(q));
    }
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      auto y1 = (Rat *)rest(y);

      mpz_t xnum_yden;
      mpz_init(xnum_yden);
      mpz_mul(xnum_yden, mpq_numref(x1->val), mpq_denref(y1->val));

      mpz_t xden_ynum;
      mpz_init(xden_ynum);
      mpz_mul(xden_ynum, mpq_denref(x1->val), mpq_numref(y1->val));

      Rat q;
      mpq_init(q.val);
      mpz_fdiv_q(mpq_numref(q.val), xnum_yden, xden_ynum);
      a = tag(tag(x), intern(q));

      mpz_clear(xden_ynum);
      mpz_clear(xnum_yden);
      return a;
    }
    }
    break;
  case term::DivT:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int: {
      Int q;
      mpz_init(q.val);
      mpz_tdiv_q(q.val, ((Int *)rest(x))->val, ((Int *)rest(y))->val);
      return tag(term::Int, intern(q));
    }
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      auto y1 = (Rat *)rest(y);

      mpz_t xnum_yden;
      mpz_init(xnum_yden);
      mpz_mul(xnum_yden, mpq_numref(x1->val), mpq_denref(y1->val));

      mpz_t xden_ynum;
      mpz_init(xden_ynum);
      mpz_mul(xden_ynum, mpq_denref(x1->val), mpq_numref(y1->val));

      Rat q;
      mpq_init(q.val);
      mpz_tdiv_q(mpq_numref(q.val), xnum_yden, xden_ynum);
      a = tag(tag(x), intern(q));

      mpz_clear(xden_ynum);
      mpz_clear(xnum_yden);
      return a;
    }
    }
    break;
  case term::Eq:
    if (x == y)
      return term::True;
    if (!isConst(x))
      break;
    if (isConst(y))
      return term::False;
    break;
  case term::Floor:
    switch (tag(x)) {
    case term::Int:
      return x;
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      Rat q;
      mpq_init(q.val);
      mpz_fdiv_q(mpq_numref(q.val), mpq_numref(x1->val), mpq_denref(x1->val));
      return tag(tag(x), intern(q));
    }
    }
    break;
  case term::IsInt: {
    if (typeof(x) == type::Int)
      return term::True;
    if (!isConst(x))
      break;
    auto x1 = (Rat *)rest(x);
    return (term)(!mpz_cmp_ui(mpq_denref(x1->val), 1));
  }
  case term::IsRat:
    // if the predicate is applied to a number of type integer or rational, then
    // of course it is a rational number
    if (typeof(x) != type::Real)
      return term::True;
    // if it is applied to a numeric constant, which must be a rational number
    // because that is the only supported format for numeric constants, then it
    // is a rational number
    if (tag(x) == term::Real)
      return term::True;
    // otherwise, we don't know
    break;
  case term::Le:
    if (x == y)
      return term::True;
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int:
      return (term)(mpz_cmp(((Int *)rest(x))->val, ((Int *)rest(y))->val) <= 0);
    case term::Rat:
    case term::Real:
      return (term)(mpq_cmp(((Rat *)rest(x))->val, ((Rat *)rest(y))->val) <= 0);
    }
    break;
  case term::Lt:
    if (x == y)
      return term::False;
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int:
      return (term)(mpz_cmp(((Int *)rest(x))->val, ((Int *)rest(y))->val) < 0);
    case term::Rat:
    case term::Real:
      return (term)(mpq_cmp(((Rat *)rest(x))->val, ((Rat *)rest(y))->val) < 0);
    }
    break;
  case term::Minus:
    switch (tag(x)) {
    case term::Int: {
      Int r;
      mpz_init(r.val);
      mpz_neg(r.val, ((Int *)rest(x))->val);
      return tag(term::Int, intern(r));
    }
    case term::Rat:
    case term::Real: {
      Rat r;
      mpq_init(r.val);
      mpq_neg(r.val, ((Rat *)rest(x))->val);
      return tag(tag(x), intern(r));
    }
    }
    break;
  case term::Mul:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int: {
      Int r;
      mpz_init(r.val);
      mpz_mul(r.val, ((Int *)rest(x))->val, ((Int *)rest(y))->val);
      return tag(term::Int, intern(r));
    }
    case term::Rat:
    case term::Real: {
      Rat r;
      mpq_init(r.val);
      mpq_mul(r.val, ((Rat *)rest(x))->val, ((Rat *)rest(y))->val);
      return tag(tag(x), intern(r));
    }
    }
    break;
  case term::RemE:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int: {
      Int r;
      mpz_init(r.val);
      mpz_ediv_r(r.val, ((Int *)rest(x))->val, ((Int *)rest(y))->val);
      return tag(term::Int, intern(r));
    }
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      auto y1 = (Rat *)rest(y);

      mpz_t xnum_yden;
      mpz_init(xnum_yden);
      mpz_mul(xnum_yden, mpq_numref(x1->val), mpq_denref(y1->val));

      mpz_t xden_ynum;
      mpz_init(xden_ynum);
      mpz_mul(xden_ynum, mpq_denref(x1->val), mpq_numref(y1->val));

      Rat r;
      mpq_init(r.val);
      mpz_ediv_r(mpq_numref(r.val), xnum_yden, xden_ynum);
      a = tag(tag(x), intern(r));

      mpz_clear(xden_ynum);
      mpz_clear(xnum_yden);
      return a;
    }
    }
    break;
  case term::RemF:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int: {
      Int r;
      mpz_init(r.val);
      mpz_fdiv_r(r.val, ((Int *)rest(x))->val, ((Int *)rest(y))->val);
      return tag(term::Int, intern(r));
    }
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      auto y1 = (Rat *)rest(y);

      mpz_t xnum_yden;
      mpz_init(xnum_yden);
      mpz_mul(xnum_yden, mpq_numref(x1->val), mpq_denref(y1->val));

      mpz_t xden_ynum;
      mpz_init(xden_ynum);
      mpz_mul(xden_ynum, mpq_denref(x1->val), mpq_numref(y1->val));

      Rat r;
      mpq_init(r.val);
      mpz_fdiv_r(mpq_numref(r.val), xnum_yden, xden_ynum);
      a = tag(tag(x), intern(r));

      mpz_clear(xden_ynum);
      mpz_clear(xnum_yden);
      return a;
    }
    }
    break;
  case term::RemT:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int: {
      Int r;
      mpz_init(r.val);
      mpz_tdiv_r(r.val, ((Int *)rest(x))->val, ((Int *)rest(y))->val);
      return tag(term::Int, intern(r));
    }
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      auto y1 = (Rat *)rest(y);

      mpz_t xnum_yden;
      mpz_init(xnum_yden);
      mpz_mul(xnum_yden, mpq_numref(x1->val), mpq_denref(y1->val));

      mpz_t xden_ynum;
      mpz_init(xden_ynum);
      mpz_mul(xden_ynum, mpq_denref(x1->val), mpq_numref(y1->val));

      Rat r;
      mpq_init(r.val);
      mpz_tdiv_r(mpq_numref(r.val), xnum_yden, xden_ynum);
      a = tag(tag(x), intern(r));

      mpz_clear(xden_ynum);
      mpz_clear(xnum_yden);
      return a;
    }
    }
    break;
  case term::Round:
    switch (tag(x)) {
    case term::Int:
      return x;
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      Rat q;
      mpq_init(q.val);
      round(mpq_numref(q.val), mpq_numref(x1->val), mpq_denref(x1->val));
      return tag(tag(x), intern(q));
    }
    }
    break;
  case term::Sub:
    if (!isConst(x))
      break;
    switch (tag(y)) {
    case term::Int: {
      Int r;
      mpz_init(r.val);
      mpz_sub(r.val, ((Int *)rest(x))->val, ((Int *)rest(y))->val);
      return tag(term::Int, intern(r));
    }
    case term::Rat:
    case term::Real: {
      Rat r;
      mpq_init(r.val);
      mpq_sub(r.val, ((Rat *)rest(x))->val, ((Rat *)rest(y))->val);
      return tag(tag(x), intern(r));
    }
    }
    break;
  case term::ToInt: {
    if (typeof(x) == type::Int)
      return x;
    if (!isConst(x))
      break;
    auto x1 = (Rat *)rest(x);
    Int q;
    mpz_init(q.val);
    mpz_fdiv_q(q.val, mpq_numref(x1->val), mpq_denref(x1->val));
    return tag(term::Int, intern(q));
  }
  case term::ToRat:
    if (typeof(x) == type::Rat)
      return x;
    switch (tag(x)) {
    case term::Int: {
      Rat r;
      mpq_init(r.val);
      mpz_set(mpq_numref(r.val), ((Int *)rest(x))->val);
      return tag(term::Rat, intern(r));
    }
    case term::Real:
      return tag(term::Rat, ((Rat *)rest(x)));
    }
    break;
  case term::ToReal:
    if (typeof(x) == type::Real)
      return x;
    switch (tag(x)) {
    case term::Int: {
      Rat r;
      mpq_init(r.val);
      mpz_set(mpq_numref(r.val), ((Int *)rest(x))->val);
      return tag(term::Real, intern(r));
    }
    case term::Rat:
      return tag(term::Real, ((Rat *)rest(x)));
    }
    break;
  case term::Trunc:
    switch (tag(x)) {
    case term::Int:
      return x;
    case term::Rat:
    case term::Real: {
      auto x1 = (Rat *)rest(x);
      Rat q;
      mpq_init(q.val);
      mpz_tdiv_q(mpq_numref(q.val), mpq_numref(x1->val), mpq_denref(x1->val));
      return tag(tag(x), intern(q));
    }
    }
    break;
  }
  return a;
}
