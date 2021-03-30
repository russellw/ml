#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

// slow tests only run when needed
#ifdef DEBUG
namespace {
const int intRange = 3;
vec<pair<term, term>> env;
unordered_map<term, int *> tables;
unordered_map<term, term> results;
vec<term> skolems;

bool eq(term a, term b) {
  vec<pair<term, term>> unified;
  return unify(a, b, unified) && !unified.n;
}

term eqv(term a, term b) { return mk(term::Eqv, a, b); }

term all(term x, term a) { return mk(term::All, a, x); }

term exists(term x, term a) { return mk(term::Exists, a, x); }

term mkInt(int val) {
  Int x;
  mpz_init_set_si(x.val, val);
  return tag(term::Int, intern(x));
}

term mkRat(int n, int d = 1) {
  Rat x;
  mpq_init(x.val);
  mpq_set_si(x.val, n, d);
  return tag(term::Rat, intern(x));
}

term mkReal(double val) {
  Rat x;
  mpq_init(x.val);
  mpq_set_d(x.val, val);
  return tag(term::Real, intern(x));
}

term eval(term a) {
  ck(a);
  switch (tag(a)) {
  case term::All: {
    assert(size(a) == 2);
    auto x = at(a, 1);
    auto body = at(a, 0);
    auto old = env.n;
    assert(typeof(x) == type::Int);
    for (int i = 0; i < intRange; ++i) {
      env.push_back(make_pair(x, mkInt(i)));
      auto r = eval(body);
      env.n = old;
      if (r == term::False)
        return term::False;
    }
    return term::True;
  }
  case term::Call: {
    auto f = at(a, 0);
    auto table = tables[f];
    assert(table);

    auto t = internType(type::Bool, type::Int);
    if (typeof(f) == t) {
      auto x = eval(at(a, 1));
      assert(tag(x) == term::Int);
      auto x1 = mpz_get_ui(((Int *)rest(x))->val);
      assert(x1 < intRange);

      return term(table[x1]);
    }

    t = internType(type::Bool, type::Int, type::Int);
    if (typeof(f) == t) {
      auto x = eval(at(a, 1));
      assert(tag(x) == term::Int);
      auto x1 = mpz_get_ui(((Int *)rest(x))->val);
      assert(x1 < intRange);

      auto y = eval(at(a, 2));
      assert(tag(y) == term::Int);
      auto y1 = mpz_get_ui(((Int *)rest(y))->val);
      assert(y1 < intRange);

      return term(table[x1 * intRange + y1]);
    }

    t = internType(type::Int, type::Int);
    if (typeof(f) == t) {
      auto x = eval(at(a, 1));
      assert(tag(x) == term::Int);
      auto x1 = mpz_get_ui(((Int *)rest(x))->val);
      assert(x1 < intRange);

      return mkInt(table[x1]);
    }

    t = internType(type::Int, type::Int, type::Int);
    if (typeof(f) == t) {
      auto x = eval(at(a, 1));
      assert(tag(x) == term::Int);
      auto x1 = mpz_get_ui(((Int *)rest(x))->val);
      assert(x1 < intRange);

      auto y = eval(at(a, 2));
      assert(tag(y) == term::Int);
      auto y1 = mpz_get_ui(((Int *)rest(y))->val);
      assert(y1 < intRange);

      return mkInt(table[x1 * intRange + y1]);
    }

    unreachable;
  }
  case term::Exists: {
    assert(size(a) == 2);
    auto x = at(a, 1);
    auto body = at(a, 0);
    auto old = env.n;
    assert(typeof(x) == type::Int);
    for (int i = 0; i < intRange; ++i) {
      env.push_back(make_pair(x, mkInt(i)));
      auto r = eval(body);
      env.n = old;
      if (r == term::True)
        return term::True;
    }
    return term::False;
  }
  case term::Sym:
  case term::Var:
    for (auto i = env.rbegin(), e = env.rend(); i != e; ++i)
      if (i->first == a)
        return i->second;
    unreachable;
  }
  if (!isCompound(a))
    return a;
  auto n = size(a);
  vec<term> v(n);
  for (si i = 0; i != n; ++i)
    v[i] = eval(at(a, i));
  a = intern(tag(a), v);
  term x;
  if (n)
    x = v[0];
  term y;
  if (n > 1)
    y = v[1];
  switch (tag(a)) {
  case term::Add:
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
  case term::And:
    for (si i = 0; i != n; ++i)
      if (at(a, i) == term::False)
        return term::False;
    return term::True;
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
  case term::Eqv:
    if (x == y)
      return term::True;
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
  case term::Imp:
    return term(x == term::False || y == term::True);
  case term::IsInt: {
    if (typeof(x) == type::Int)
      return term::True;
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
    return term::True;
  case term::Le:
    if (x == y)
      return term::True;
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
  case term::Not:
    switch (x) {
    case term::False:
      return term::True;
    case term::True:
      return term::False;
    }
    break;
  case term::Or:
    for (si i = 0; i != n; ++i)
      if (at(a, i) == term::True)
        return term::True;
    return term::False;
  case term::RemE:
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
  unreachable;
  return term::Floor;
}

term fn(type t, sym *s) {
  if (s->t == type::none)
    s->t = t;
  assert(s->t == t);
  return tag(term::Sym, s);
}

vec<term> fns;
vec<term> vars;

void getFns(term a) {
  if (tag(a) == term::Sym && find(fns.p, fns.end(), a) == fns.end()) {
    fns.push_back(a);
    auto s = (sym *)rest(a);
    if (!*s->v)
      skolems.push_back(a);
  }
  if (isCompound(a))
    for (auto b : a)
      getFns(b);
}

bool satClause(clause *c) {
  for (auto i = c->v, e = c->v + c->nn; i != e; ++i)
    if (eval(*i) == term::False)
      return true;
  for (auto i = c->v + c->nn, e = c->v + c->n; i != e; ++i)
    if (eval(*i) == term::True)
      return true;
  return false;
}

bool satProblem() {
  for (auto c : problem)
    if (!satClause(c))
      return false;
  return true;
}

bool satVar(si i) {
  if (i < vars.n) {
    auto a = vars[i];
    auto old = env.n;
    switch (typeof(a)) {
    case type::Bool: {
      env.push_back(make_pair(a, term::False));
      auto r = satVar(i + 1);
      env.n = old;
      if (!r)
        return false;

      env.push_back(make_pair(a, term::True));
      r = satVar(i + 1);
      env.n = old;
      if (!r)
        return false;

      return true;
    }
    case type::Int: {
      for (int j = 0; j < intRange; ++j) {
        env.push_back(make_pair(a, mkInt(j)));
        auto r = satVar(i + 1);
        env.n = old;
        if (!r)
          return false;
      }
      return true;
    }
    }
    unreachable;
  }
  return satProblem();
}

bool satFn(si i);

bool satPredInts(si fi, int *table, si n, si i) {
  if (i < n) {
    for (int j = 0; j < 2; ++j) {
      table[i] = j;
      if (satPredInts(fi, table, n, i + 1))
        return true;
    }
    return false;
  }
  return satFn(fi + 1);
}

bool satFnInts(si fi, int *table, si n, si i) {
  if (i < n) {
    for (int j = 0; j < intRange; ++j) {
      table[i] = j;
      if (satFnInts(fi, table, n, i + 1))
        return true;
    }
    return false;
  }
  return satFn(fi + 1);
}

bool satFn(si i) {
  if (i < fns.n) {
    auto a = fns[i];
    auto old = env.n;
    switch (typeof(a)) {
    case type::Bool: {
      env.push_back(make_pair(a, term::False));
      auto r = satFn(i + 1);
      env.n = old;
      if (r) {
        results[a] = term::False;
        return true;
      }

      env.push_back(make_pair(a, term::True));
      r = satFn(i + 1);
      env.n = old;
      if (r) {
        results[a] = term::True;
        return true;
      }

      return false;
    }
    case type::Int: {
      for (int j = 0; j < intRange; ++j) {
        env.push_back(make_pair(a, mkInt(j)));
        auto r = satFn(i + 1);
        env.n = old;
        if (r) {
          results[a] = mkInt(j);
          return true;
        }
      }
      return false;
    }
    }

    auto t = internType(type::Bool, type::Int);
    if (typeof(a) == t) {
      auto n = intRange;
      auto &table = tables[a];
      if (!table)
        table = new int[n];
      auto r = satPredInts(i, table, n, 0);
      return r;
    }

    t = internType(type::Bool, type::Int, type::Int);
    if (typeof(a) == t) {
      auto n = intRange * intRange;
      auto &table = tables[a];
      if (!table)
        table = new int[n];
      auto r = satPredInts(i, table, n, 0);
      return r;
    }

    t = internType(type::Int, type::Int);
    if (typeof(a) == t) {
      auto n = intRange;
      auto &table = tables[a];
      if (!table)
        table = new int[n];
      auto r = satFnInts(i, table, n, 0);
      return r;
    }

    t = internType(type::Int, type::Int, type::Int);
    if (typeof(a) == t) {
      auto n = intRange * intRange;
      auto &table = tables[a];
      if (!table)
        table = new int[n];
      auto r = satFnInts(i, table, n, 0);
      return r;
    }

    debug(a);
    debug(typeof(a));
    unreachable;
  }
  return satVar(0);
}

bool sat() {
  fns.n = 0;
  vars.n = 0;
  skolems.n = 0;
  results.clear();
  for (auto c : problem)
    for (auto i = c->v, e = c->v + c->n; i != e; ++i) {
      getFns(*i);
      getFreeVars(*i, vars);
    }
  return satFn(0);
}

void setFormula(term a) {
  problem.n = 0;
  auto c = mk(a, how::none);
  problem.push_back(c);
}

void testSat(term a) {
  setFormula(a);
  auto b = sat();
  problem.n = 0;
  initClauses();
  cnf(a, 0);
  assert(b == sat());
}

void testSat(term a, bool expected) {
  setFormula(a);
  auto b = sat();
  problem.n = 0;
  initClauses();
  cnf(a, 0);
  if (0) {
    debug(problem.n);
    for (auto c : problem)
      debug(c);
  }
  assert(b == sat());
  assert(b == expected);
}

int call(term f, int x) {
  auto t = typeof(f);
  assert(isCompound(t));
  auto p = tcompoundp(t);
  assert(p->n == 2);
  auto table = tables[f];
  assert(table);
  return table[x];
}

int call(term f, int x, int y) {
  auto t = typeof(f);
  assert(isCompound(t));
  auto p = tcompoundp(t);
  assert(p->n == 3);
  auto table = tables[f];
  assert(table);
  return table[x * intRange + y];
}
} // namespace

// expanded version of CNF conversion algorithm for easier debugging
namespace cnfx {
// estimate how many clauses a term will expand into, for the purpose of
// deciding when subformulas need to be renamed; the answer could exceed 2^31,
// but then we don't actually need the number, we only need to know whether it
// went over the threshold
const si many = 10;
si nclauses(bool pol, term a);

si nclausesAnd(bool pol, term a) {
  si r = 0;
  for (auto b : a) {
    r += nclauses(pol, b);
    if (r >= many)
      return many;
  }
  return r;
}

si nclausesOr(bool pol, term a) {
  si r = 1;
  for (auto b : a) {
    r *= nclauses(pol, b);
    if (r >= many)
      return many;
  }
  return r;
}

si nclauses(bool pol, term a) {
  switch (tag(a)) {
  case term::All:
  case term::Exists:
  case term::Not:
    return nclauses(pol, at(a, 0));
  case term::And:
    return pol ? nclausesAnd(pol, a) : nclausesOr(pol, a);
  case term::Eqv: {
    auto x = at(a, 0);
    auto x0 = nclauses(0, x);
    if (x0 >= many - 1)
      return many;
    auto x1 = nclauses(1, x);
    if (x1 >= many - 1)
      return many;
    auto y = at(a, 1);
    auto y0 = nclauses(0, y);
    if (y0 >= many - 1)
      return many;
    auto y1 = nclauses(1, y);
    auto r = pol ? x0 * y1 + x1 * y0 : x0 * y0 + x1 * y1;
    if (r >= many)
      return many;
    return r;
  }
  case term::Or:
    return pol ? nclausesOr(pol, a) : nclausesAnd(pol, a);
  }
  return 1;
}

// creating new functions is necessary both to skolemize existential variables
// and to rename subformulas to avoid exponential blowup
term skolem(type rt, const vec<term> &vars) {
  // atom
  auto n = vars.n;
  if (!n)
    return tag(term::Sym, mkSym(rt));

  // compound type
  vec<type> v(n + 1);
  v[0] = rt;
  for (si i = 0; i != n; ++i)
    v[i + 1] = varType(vars[i]);

  // compound
  auto r = mk(n + 1);
  *r->v = tag(term::Sym, mkSym(internType(v)));
  memcpy(r->v + 1, vars.p, n * sizeof(term));
  return tag(term::Call, r);
}

// rename subformulas to avoid exponential blowup
term nnf(term a);

term renameEqv(term a) {
  a = nnf(a);
  vec<term> freeVars;
  getFreeVars(a, freeVars);
  auto b = skolem(type::Bool, freeVars);
  cnf(mk(term::And, mk(term::Imp, b, a), mk(term::Imp, a, b)), 0);
  return b;
}

// negation normal form
// for-all vars map to fresh vars
// exists vars map to skolem functions
struct quant {
  bool exists;
  term var;
  term renamed;

  quant() {}

  quant(bool exists, term var, term renamed)
      : exists(exists), var(var), renamed(renamed) {}
};

term nnf(vec<quant> &boundVars, unordered_map<type, si> &newVars, bool pol,
         term a);

term all(vec<quant> &boundVars, unordered_map<type, si> &newVars, bool pol,
         term a) {
  auto n = size(a);
  auto old = boundVars.n;
  boundVars.resize(old + n - 1);
  for (si i = 1; i != n; ++i) {
    auto x = at(a, i);
    auto t = varType(x);
    boundVars[old + i - 1] = quant(0, x, var(t, newVars[t]++));
  }
  a = nnf(boundVars, newVars, pol, at(a, 0));
  boundVars.n = old;
  return a;
}

term exists(vec<quant> &boundVars, unordered_map<type, si> &newVars, bool pol,
            term a) {
  auto n = size(a);
  auto old = boundVars.n;
  boundVars.resize(old + n - 1);
  for (si i = 1; i != n; ++i) {
    vec<term> allVars;
    for (auto j = boundVars.p, e = boundVars.p + old; j != e; ++j)
      if (!j->exists)
        allVars.push_back(j->renamed);
    auto x = at(a, i);
    auto y = skolem(varType(x), allVars);
    boundVars[old + i - 1] = quant(1, x, y);
  }
  a = nnf(boundVars, newVars, pol, at(a, 0));
  boundVars.n = old;
  return a;
}

term nnf(vec<quant> &boundVars, unordered_map<type, si> &newVars, bool pol,
         term a) {
  switch (tag(a)) {
  case term::All:
    return pol ? all(boundVars, newVars, pol, a)
               : exists(boundVars, newVars, pol, a);
  case term::And:
    if (pol) {
      vec<term> v;
      for (auto b : a) {
        b = nnf(boundVars, newVars, pol, b);
        if (tag(b) == term::And)
          v.insert(v.end(), begin(b), end(b));
        else
          v.push_back(b);
      }
      return mk(term::And, v);
    } else {
      vec<term> v;
      for (auto b : a)
        v.push_back(nnf(boundVars, newVars, pol, b));
      return mk(term::Or, v);
    }
  case term::Eqv: {
    auto x = at(a, 0);
    if (nclauses(0, x) >= many || nclauses(1, x) >= many)
      x = renameEqv(x);
    auto y = at(a, 1);
    if (nclauses(0, y) >= many || nclauses(1, y) >= many)
      y = renameEqv(y);
    return nnf(boundVars, newVars, pol,
               mk(term::And, mk(term::Imp, x, y), mk(term::Imp, y, x)));
  }
  case term::Exists:
    return pol ? exists(boundVars, newVars, pol, a)
               : all(boundVars, newVars, pol, a);
  case term::False:
    return term(pol ^ 1);
  case term::Imp: {
    auto x = at(a, 0);
    auto y = at(a, 1);
    return nnf(boundVars, newVars, pol, mk(term::Or, mk(term::Not, x), y));
  }
  case term::Not:
    return nnf(boundVars, newVars, pol ^ 1, at(a, 0));
  case term::Or:
    if (pol) {
      vec<term> v;
      for (auto b : a)
        v.push_back(nnf(boundVars, newVars, pol, b));
      return mk(term::Or, v);
    } else {
      vec<term> v;
      for (auto b : a) {
        b = nnf(boundVars, newVars, pol, b);
        if (tag(b) == term::And)
          v.insert(v.end(), begin(b), end(b));
        else
          v.push_back(b);
      }
      return mk(term::And, v);
    }
  case term::True:
    return term(pol);
  case term::Var:
    for (auto i = boundVars.rbegin(), e = boundVars.rend(); i != e; ++i)
      if (i->var == a)
        return i->renamed;
    unreachable;
  }
  if (isCompound(a)) {
    auto n = size(a);
    auto r = mk(n);
    for (si i = 0; i != n; ++i)
      r->v[i] = nnf(boundVars, newVars, 1, at(a, i));
    a = tag(tag(a), r);
  }
  return pol ? a : mk(term::Not, a);
}

term nnf(term a) {
  vec<quant> boundVars;
  unordered_map<type, si> newVars;
  vec<term> freeVars;
  getFreeVars(a, freeVars);
  for (auto x : freeVars) {
    auto t = varType(x);
    boundVars.push_back(quant(0, x, var(t, newVars[t]++)));
  }
  return nnf(boundVars, newVars, 1, a);
}

term distrib(term a) {
  switch (tag(a)) {
  case term::And: {
    vec<term> v;
    for (auto b : a) {
      b = distrib(b);
      if (tag(b) == term::And) {
        for (auto c : b)
          v.push_back(c);
        continue;
      }
      v.push_back(b);
    }
    return mk(term::And, v);
  }
  case term::Or: {
    vec<term> ands;
    for (auto b : a) {
      b = distrib(b);
      ands.push_back(b);
    }

    // a vector of indexes into And terms
    // that will provide a slice through the And arguments
    // to create a single Or term
    auto n = ands.n;
    vec<si> j(n);
    memset(j.p, 0, n * sizeof *j.p);

    // the components of a single Or term
    vec<term> literals(n);

    // all the Or terms
    // that will become the arguments to an And
    vec<term> ors;

    // cartesian product of Ands
    for (;;) {
      // make another Or that takes a slice through the And args
      for (si i = 0; i != n; ++i) {
        auto b = ands[i];
        if (tag(b) == term::And)
          b = at(b, j[i]);
        else
          assert(!j[i]);
        literals[i] = b;
      }
      ors.push_back(mk(term::Or, literals));

      // take the next slice
      for (si i = n;;) {
        // if we have done all the slices, return And of Ors
        if (!i)
          return mk(term::And, ors);

        // next element of the index vector
        // this is equivalent to increment with carry, of a multi-precision
        // integer, except that the 'base', the maximum value of a 'digit', is
        // different for each place, being the number of arguments to the And at
        // that position
        auto b = ands[--i];
        if (tag(b) == term::And) {
          auto m = size(b);
          if (++j[i] == m) {
            j[i] = 0;
            // carry
            continue;
          }
          break;
        }
      }
    }
  }
  }
  return a;
}

// make clauses
vec<term> neg, pos;

void toLiterals(term a) {
  switch (tag(a)) {
  case term::And:
    unreachable;
  case term::Not:
    neg.push_back(at(a, 0));
    return;
  case term::Or:
    for (auto b : a)
      toLiterals(b);
    return;
  }
  pos.push_back(a);
}

void toClause(term a) {
  assert(tag(a) != term::And);
  neg.n = pos.n = 0;
  toLiterals(a);
  input(neg, pos, how::cnf);
}

void cnf(term a, clause *f) {
  ck(a);
  a = nnf(a);
  ck(a);
  a = distrib(a);
  ck(a);
  if (tag(a) == term::And)
    for (auto b : a)
      toClause(b);
  else
    toClause(a);
}

void testSat(term a, bool expected) {
  setFormula(a);
  auto b = sat();
  problem.n = 0;
  initClauses();
  cnfx::cnf(a, 0);
  if (0) {
    debug(problem.n);
    for (auto c : problem)
      debug(c);
  }
  assert(b == sat());
  assert(b == expected);
}
} // namespace cnfx

void test1() {
  assert(eval(intern(term::Eq, mkInt(1), mkInt(1))) == term::True);
  assert(eval(intern(term::Eq, mkInt(1), mkInt(2))) == term::False);
  assert(eval(intern(term::Eq, mkRat(1, 3), mkRat(2, 6))) == term::True);

  assert(eval(intern(term::Lt, mkInt(1), mkInt(2))) == term::True);
  assert(eval(intern(term::Lt, mkInt(2), mkInt(1))) == term::False);
  assert(eval(intern(term::Lt, mkRat(1, 3), mkRat(1, 2))) == term::True);

  assert(eval(intern(term::Le, mkInt(1), mkInt(2))) == term::True);
  assert(eval(intern(term::Le, mkInt(2), mkInt(1))) == term::False);
  assert(eval(intern(term::Le, mkRat(1, 3), mkRat(1, 2))) == term::True);

  assert(eval(intern(term::Add, mkInt(1), mkInt(2))) == mkInt(3));
  assert(eval(intern(term::Add, mkRat(1, 3), mkRat(1, 2))) == mkRat(5, 6));

  assert(eval(intern(term::Sub, mkInt(10), mkInt(2))) == mkInt(8));
  assert(eval(intern(term::Sub, mkRat(1, 2), mkRat(1, 3))) == mkRat(1, 6));

  assert(eval(intern(term::Mul, mkInt(10), mkInt(2))) == mkInt(20));
  assert(eval(intern(term::Mul, mkRat(1, 2), mkRat(1, 7))) == mkRat(1, 14));

  assert(eval(intern(term::Div, mkRat(1, 2), mkRat(1, 7))) == mkRat(7, 2));

  assert(eval(intern(term::Minus, mkInt(1))) == mkInt(-1));
  assert(eval(intern(term::Minus, mkReal(-1.5))) == mkReal(1.5));

  assert(eval(intern(term::DivF, mkInt(5), mkInt(3))) == mkInt(1));
  assert(eval(intern(term::DivF, mkInt(-5), mkInt(3))) == mkInt(-2));
  assert(eval(intern(term::DivF, mkInt(5), mkInt(-3))) == mkInt(-2));
  assert(eval(intern(term::DivF, mkInt(-5), mkInt(-3))) == mkInt(1));
  assert(eval(intern(term::DivF, mkRat(5), mkRat(3))) == mkRat(1));
  assert(eval(intern(term::DivF, mkRat(-5), mkRat(3))) == mkRat(-2));
  assert(eval(intern(term::DivF, mkRat(5), mkRat(-3))) == mkRat(-2));
  assert(eval(intern(term::DivF, mkRat(-5), mkRat(-3))) == mkRat(1));

  assert(eval(intern(term::RemF, mkInt(5), mkInt(3))) == mkInt(2));
  assert(eval(intern(term::RemF, mkInt(-5), mkInt(3))) == mkInt(1));
  assert(eval(intern(term::RemF, mkInt(5), mkInt(-3))) == mkInt(-1));
  assert(eval(intern(term::RemF, mkInt(-5), mkInt(-3))) == mkInt(-2));
  assert(eval(intern(term::RemF, mkRat(5), mkRat(3))) == mkRat(2));
  assert(eval(intern(term::RemF, mkRat(-5), mkRat(3))) == mkRat(1));
  assert(eval(intern(term::RemF, mkRat(5), mkRat(-3))) == mkRat(-1));
  assert(eval(intern(term::RemF, mkRat(-5), mkRat(-3))) == mkRat(-2));

  assert(eval(intern(term::DivT, mkInt(5), mkInt(3))) == mkInt(5 / 3));
  assert(eval(intern(term::DivT, mkInt(-5), mkInt(3))) == mkInt(-5 / 3));
  assert(eval(intern(term::DivT, mkInt(5), mkInt(-3))) == mkInt(5 / -3));
  assert(eval(intern(term::DivT, mkInt(-5), mkInt(-3))) == mkInt(-5 / -3));
  assert(eval(intern(term::DivT, mkInt(5), mkInt(3))) == mkInt(1));
  assert(eval(intern(term::DivT, mkInt(-5), mkInt(3))) == mkInt(-1));
  assert(eval(intern(term::DivT, mkInt(5), mkInt(-3))) == mkInt(-1));
  assert(eval(intern(term::DivT, mkInt(-5), mkInt(-3))) == mkInt(1));
  assert(eval(intern(term::DivT, mkRat(5), mkRat(3))) == mkRat(1));
  assert(eval(intern(term::DivT, mkRat(-5), mkRat(3))) == mkRat(-1));
  assert(eval(intern(term::DivT, mkRat(5), mkRat(-3))) == mkRat(-1));
  assert(eval(intern(term::DivT, mkRat(-5), mkRat(-3))) == mkRat(1));

  assert(eval(intern(term::RemT, mkInt(5), mkInt(3))) == mkInt(5 % 3));
  assert(eval(intern(term::RemT, mkInt(-5), mkInt(3))) == mkInt(-5 % 3));
  assert(eval(intern(term::RemT, mkInt(5), mkInt(-3))) == mkInt(5 % -3));
  assert(eval(intern(term::RemT, mkInt(-5), mkInt(-3))) == mkInt(-5 % -3));
  assert(eval(intern(term::RemT, mkInt(5), mkInt(3))) == mkInt(2));
  assert(eval(intern(term::RemT, mkInt(-5), mkInt(3))) == mkInt(-2));
  assert(eval(intern(term::RemT, mkInt(5), mkInt(-3))) == mkInt(2));
  assert(eval(intern(term::RemT, mkInt(-5), mkInt(-3))) == mkInt(-2));
  assert(eval(intern(term::RemT, mkRat(5), mkRat(3))) == mkRat(2));
  assert(eval(intern(term::RemT, mkRat(-5), mkRat(3))) == mkRat(-2));
  assert(eval(intern(term::RemT, mkRat(5), mkRat(-3))) == mkRat(2));
  assert(eval(intern(term::RemT, mkRat(-5), mkRat(-3))) == mkRat(-2));

  assert(eval(intern(term::DivE, mkInt(7), mkInt(3))) == mkInt(2));
  assert(eval(intern(term::DivE, mkInt(-7), mkInt(3))) == mkInt(-3));
  assert(eval(intern(term::DivE, mkInt(7), mkInt(-3))) == mkInt(-2));
  assert(eval(intern(term::DivE, mkInt(-7), mkInt(-3))) == mkInt(3));
  assert(eval(intern(term::DivE, mkRat(7), mkRat(3))) == mkRat(2));
  assert(eval(intern(term::DivE, mkRat(-7), mkRat(3))) == mkRat(-3));
  assert(eval(intern(term::DivE, mkRat(7), mkRat(-3))) == mkRat(-2));
  assert(eval(intern(term::DivE, mkRat(-7), mkRat(-3))) == mkRat(3));

  assert(eval(intern(term::RemE, mkInt(7), mkInt(3))) == mkInt(1));
  assert(eval(intern(term::RemE, mkInt(-7), mkInt(3))) == mkInt(2));
  assert(eval(intern(term::RemE, mkInt(7), mkInt(-3))) == mkInt(1));
  assert(eval(intern(term::RemE, mkInt(-7), mkInt(-3))) == mkInt(2));
  assert(eval(intern(term::RemE, mkRat(7), mkRat(3))) == mkRat(1));
  assert(eval(intern(term::RemE, mkRat(-7), mkRat(3))) == mkRat(2));
  assert(eval(intern(term::RemE, mkRat(7), mkRat(-3))) == mkRat(1));
  assert(eval(intern(term::RemE, mkRat(-7), mkRat(-3))) == mkRat(2));

  assert(eval(intern(term::Ceil, mkInt(0))) == mkInt(0));
  assert(eval(intern(term::Ceil, mkRat(0))) == mkRat(0));
  assert(eval(intern(term::Ceil, mkRat(1, 10))) == mkRat(1));
  assert(eval(intern(term::Ceil, mkRat(5, 10))) == mkRat(1));
  assert(eval(intern(term::Ceil, mkRat(9, 10))) == mkRat(1));
  assert(eval(intern(term::Ceil, mkRat(-1, 10))) == mkRat(0));
  assert(eval(intern(term::Ceil, mkRat(-5, 10))) == mkRat(0));
  assert(eval(intern(term::Ceil, mkRat(-9, 10))) == mkRat(0));

  assert(eval(intern(term::Floor, mkInt(0))) == mkInt(0));
  assert(eval(intern(term::Floor, mkRat(0))) == mkRat(0));
  assert(eval(intern(term::Floor, mkRat(1, 10))) == mkRat(0));
  assert(eval(intern(term::Floor, mkRat(5, 10))) == mkRat(0));
  assert(eval(intern(term::Floor, mkRat(9, 10))) == mkRat(0));
  assert(eval(intern(term::Floor, mkRat(-1, 10))) == mkRat(-1));
  assert(eval(intern(term::Floor, mkRat(-5, 10))) == mkRat(-1));
  assert(eval(intern(term::Floor, mkRat(-9, 10))) == mkRat(-1));

  assert(eval(intern(term::Trunc, mkInt(0))) == mkInt(0));
  assert(eval(intern(term::Trunc, mkRat(0))) == mkRat(0));
  assert(eval(intern(term::Trunc, mkRat(1, 10))) == mkRat(0));
  assert(eval(intern(term::Trunc, mkRat(5, 10))) == mkRat(0));
  assert(eval(intern(term::Trunc, mkRat(9, 10))) == mkRat(0));
  assert(eval(intern(term::Trunc, mkRat(-1, 10))) == mkRat(0));
  assert(eval(intern(term::Trunc, mkRat(-5, 10))) == mkRat(0));
  assert(eval(intern(term::Trunc, mkRat(-9, 10))) == mkRat(0));

  assert(eval(intern(term::Round, mkInt(0))) == mkInt(0));
  assert(eval(intern(term::Round, mkRat(0))) == mkRat(0));
  assert(eval(intern(term::Round, mkRat(1, 10))) == mkRat(0));
  assert(eval(intern(term::Round, mkRat(5, 10))) == mkRat(0));
  assert(eval(intern(term::Round, mkRat(9, 10))) == mkRat(1));
  assert(eval(intern(term::Round, mkRat(-1, 10))) == mkRat(0));
  assert(eval(intern(term::Round, mkRat(-5, 10))) == mkRat(0));
  assert(eval(intern(term::Round, mkRat(-9, 10))) == mkRat(-1));
  assert(eval(intern(term::Round, mkRat(15, 10))) == mkRat(2));
  assert(eval(intern(term::Round, mkRat(25, 10))) == mkRat(2));
  assert(eval(intern(term::Round, mkRat(35, 10))) == mkRat(4));
  assert(eval(intern(term::Round, mkRat(45, 10))) == mkRat(4));

  assert(eval(intern(term::IsInt, mkRat(5, 5))) == term::True);
  assert(eval(intern(term::IsInt, mkRat(5, 10))) == term::False);

  assert(eval(intern(term::IsRat, mkRat(45, 10))) == term::True);
  assert(eval(intern(term::IsRat, mkReal(2.5))) == term::True);

  assert(eval(intern(term::ToInt, mkInt(0))) == mkInt(0));
  assert(eval(intern(term::ToInt, mkRat(0))) == mkInt(0));
  assert(eval(intern(term::ToInt, mkRat(1, 10))) == mkInt(0));
  assert(eval(intern(term::ToInt, mkRat(5, 10))) == mkInt(0));
  assert(eval(intern(term::ToInt, mkRat(9, 10))) == mkInt(0));
  assert(eval(intern(term::ToInt, mkRat(-1, 10))) == mkInt(-1));
  assert(eval(intern(term::ToInt, mkRat(-5, 10))) == mkInt(-1));
  assert(eval(intern(term::ToInt, mkRat(-9, 10))) == mkInt(-1));

  assert(eval(intern(term::ToRat, mkInt(7))) == mkRat(7));
  assert(eval(intern(term::ToRat, mkRat(7))) == mkRat(7));
  assert(eval(intern(term::ToRat, mkReal(7.0))) == mkRat(7));

  assert(eval(intern(term::ToReal, mkInt(7))) == mkReal(7));
  assert(eval(intern(term::ToReal, mkRat(7))) == mkReal(7));
  assert(eval(intern(term::ToReal, mkReal(7.0))) == mkReal(7));

  initSyms();
  auto a = fn(type::Int, intern("a"));
  auto b = fn(type::Int, intern("b"));

  env.push_back(make_pair(a, mkInt(1)));
  env.push_back(make_pair(b, mkInt(2)));

  assert(eval(intern(term::Eq, a, mkInt(1))) == term::True);
  assert(eval(intern(term::Eq, b, mkInt(2))) == term::True);

  assert(eval(intern(term::Not, term::False)) == term::True);
  assert(eval(intern(term::Not, term::True)) == term::False);

  assert(eval(intern(term::Eqv, term::False, term::True)) == term::False);
  assert(eval(intern(term::Eqv, term::True, term::False)) == term::False);
  assert(eval(intern(term::Eqv, term::False, term::False)) == term::True);
  assert(eval(intern(term::Eqv, term::True, term::True)) == term::True);

  assert(eval(intern(term::And, term::False, term::False, term::False)) ==
         term::False);
  assert(eval(intern(term::And, term::False, term::False, term::True)) ==
         term::False);
  assert(eval(intern(term::And, term::False, term::True, term::False)) ==
         term::False);
  assert(eval(intern(term::And, term::False, term::True, term::True)) ==
         term::False);
  assert(eval(intern(term::And, term::True, term::False, term::False)) ==
         term::False);
  assert(eval(intern(term::And, term::True, term::False, term::True)) ==
         term::False);
  assert(eval(intern(term::And, term::True, term::True, term::False)) ==
         term::False);
  assert(eval(intern(term::And, term::True, term::True, term::True)) ==
         term::True);

  assert(eval(intern(term::Or, term::False, term::False, term::False)) ==
         term::False);
  assert(eval(intern(term::Or, term::False, term::False, term::True)) ==
         term::True);
  assert(eval(intern(term::Or, term::False, term::True, term::False)) ==
         term::True);
  assert(eval(intern(term::Or, term::False, term::True, term::True)) ==
         term::True);
  assert(eval(intern(term::Or, term::True, term::False, term::False)) ==
         term::True);
  assert(eval(intern(term::Or, term::True, term::False, term::True)) ==
         term::True);
  assert(eval(intern(term::Or, term::True, term::True, term::False)) ==
         term::True);
  assert(eval(intern(term::Or, term::True, term::True, term::True)) ==
         term::True);

  testSat(term::True, true);
  testSat(term::False, false);
  // duplicate is intentional
  testSat(term::False, false);

  testSat(intern(term::Eq, a, mkInt(1)), true);
  testSat(intern(term::Eq, b, mkInt(2)), true);
  testSat(intern(term::Eq, b, mkInt(intRange)), false);

  testSat(intern(term::Not, term::False), true);
  testSat(intern(term::Not, term::True), false);

  testSat(intern(term::And, term::False, term::False, term::False), false);
  testSat(intern(term::And, term::False, term::False, term::True), false);
  testSat(intern(term::And, term::False, term::True, term::False), false);
  testSat(intern(term::And, term::False, term::True, term::True), false);
  testSat(intern(term::And, term::True, term::False, term::False), false);
  testSat(intern(term::And, term::True, term::False, term::True), false);
  testSat(intern(term::And, term::True, term::True, term::False), false);
  testSat(intern(term::And, term::True, term::True, term::True), true);

  testSat(intern(term::Or, term::False, term::False, term::False), false);
  testSat(intern(term::Or, term::False, term::False, term::True), true);
  testSat(intern(term::Or, term::False, term::True, term::False), true);
  testSat(intern(term::Or, term::False, term::True, term::True), true);
  testSat(intern(term::Or, term::True, term::False, term::False), true);
  testSat(intern(term::Or, term::True, term::False, term::True), true);
  testSat(intern(term::Or, term::True, term::True, term::False), true);
  testSat(intern(term::Or, term::True, term::True, term::True), true);

  testSat(intern(term::Eqv, term::False, term::True), false);
  testSat(intern(term::Eqv, term::True, term::False), false);
  testSat(intern(term::Eqv, term::False, term::False), true);
  testSat(intern(term::Eqv, term::True, term::True), true);

  auto p = fn(type::Bool, intern("p"));
  auto q = fn(type::Bool, intern("q"));

  testSat(p, true);
  testSat(intern(term::Not, p), true);
  testSat(intern(term::And, p, p), true);
  testSat(intern(term::And, p, intern(term::Not, p)), false);
  testSat(intern(term::And, p, intern(term::Not, q)), true);
  testSat(intern(term::Or, p, p), true);
  testSat(intern(term::Or, p, intern(term::Not, p)), true);
  testSat(intern(term::Or, p, intern(term::Not, q)), true);
  testSat(intern(term::Eqv, p, p), true);
  testSat(intern(term::Eqv, p, q), true);
  testSat(intern(term::Eqv, p, intern(term::Not, p)), false);
  testSat(intern(term::Imp, p, p), true);
  testSat(intern(term::Imp, p, intern(term::Not, p)), true);
  testSat(intern(term::Imp, p, q), true);

  testSat(intern(term::Eq, intern(term::Add, a, mkInt(1)), mkInt(1)), true);
  testSat(intern(term::Eq, intern(term::Add, a, mkInt(1)), mkInt(2)), true);

  auto p1 = fn(type::Bool, intern("p1"));
  auto p2 = fn(type::Bool, intern("p2"));
  auto p3 = fn(type::Bool, intern("p3"));
  auto p4 = fn(type::Bool, intern("p4"));
  auto p5 = fn(type::Bool, intern("p5"));

  testSat(eqv(p1, eqv(p2, eqv(p1, p2))), true);
  testSat(eqv(p1, eqv(p2, eqv(p3, eqv(p1, eqv(p2, p3))))), true);
  testSat(eqv(p1, eqv(p2, eqv(p3, eqv(p4, eqv(p1, eqv(p2, eqv(p3, p4))))))),
          true);
  testSat(
      eqv(p1,
          eqv(p2,
              eqv(p3,
                  eqv(p4, eqv(p4, eqv(p5, eqv(p2, eqv(p3, eqv(p4, p5))))))))),
      true);

  auto x = var(type::Int, 0);
  auto y = var(type::Int, 1);
  auto z = var(type::Int, 2);

  testSat(all(x, intern(term::Eq, x, mkInt(1))), false);
  testSat(all(x, all(y, intern(term::Eq, x, x))), true);
  testSat(all(x, all(y, intern(term::And, intern(term::Eq, x, x),
                               intern(term::Eq, y, y)))),
          true);
  testSat(all(x, all(y, intern(term::Eq, x, y))), false);
  testSat(all(x, all(y, intern(term::And, intern(term::Eq, x, y),
                               intern(term::Eq, y, x)))),
          false);
  testSat(exists(x, intern(term::Eq, x, mkInt(1))), true);
  testSat(
      exists(x, exists(y, intern(term::Eq, intern(term::Add, x, y), mkInt(1)))),
      true);
  testSat(
      exists(x, exists(y, exists(z, intern(term::And, intern(term::Lt, x, y),
                                           intern(term::Lt, y, z))))),
      true);
  testSat(exists(x, all(y, intern(term::Eq, x, y))), false);

  // searching for functions
  auto f1 = fn(internType(type::Int, type::Int), intern("f1"));
  testSat(all(x, intern(term::Eq, intern(term::Call, f1, x), mkInt(2))), true);

  testSat(all(x, exists(y, intern(term::Eq, x, y))), true);

  auto f2 = fn(internType(type::Int, type::Int, type::Int), intern("f2"));
  testSat(
      all(x, all(y, intern(term::Eq, intern(term::Call, f2, x, y), mkInt(2)))),
      true);

  auto q1 = fn(internType(type::Bool, type::Int), intern("q1"));
  testSat(all(x, intern(term::Call, q1, x)), true);

  auto q2 = fn(internType(type::Bool, type::Int, type::Int), intern("q2"));
  testSat(all(x, all(y, intern(term::Call, q2, x, y))), true);

  // returning results
  // a=1
  testSat(intern(term::Eq, a, mkInt(1)), true);
  assert(results[a] == mkInt(1));

  // q1(x)
  testSat(all(x, intern(term::Call, q1, x)), true);
  assert(call(q1, 0) == 1);
  assert(call(q1, 1) == 1);
  assert(call(q1, 2) == 1);

  // f1(x)=2
  testSat(all(x, intern(term::Eq, intern(term::Call, f1, x), mkInt(2))), true);
  assert(call(f1, 0) == 2);
  assert(call(f1, 1) == 2);
  assert(call(f1, 2) == 2);

  // q2(x,y)
  testSat(all(x, all(y, intern(term::Call, q2, x, y))), true);
  assert(call(q2, 0, 0) == 1);
  assert(call(q2, 0, 1) == 1);
  assert(call(q2, 0, 2) == 1);
  assert(call(q2, 1, 0) == 1);
  assert(call(q2, 1, 1) == 1);
  assert(call(q2, 1, 2) == 1);
  assert(call(q2, 2, 0) == 1);
  assert(call(q2, 2, 1) == 1);
  assert(call(q2, 2, 2) == 1);

  // q1(x) <=> x=2
  testSat(
      all(x, mk(term::Eqv, mk(term::Call, q1, x), mk(term::Eq, x, mkInt(2)))),
      true);
  assert(call(q1, 0) == 0);
  assert(call(q1, 1) == 0);
  assert(call(q1, 2) == 1);

  // q2(x,y) <=> x=y
  testSat(all(x, all(y, mk(term::Eqv, mk(term::Call, q2, x, y),
                           mk(term::Eq, x, y)))),
          true);
  assert(call(q2, 0, 0) == 1);
  assert(call(q2, 0, 1) == 0);
  assert(call(q2, 0, 2) == 0);
  assert(call(q2, 1, 0) == 0);
  assert(call(q2, 1, 1) == 1);
  assert(call(q2, 1, 2) == 0);
  assert(call(q2, 2, 0) == 0);
  assert(call(q2, 2, 1) == 0);
  assert(call(q2, 2, 2) == 1);

  testSat(exists(x, intern(term::Eq, x, mkInt(1))), true);
  assert(skolems.n == 1);
  auto sk = skolems[0];
  assert(typeof(sk) == type::Int);
  assert(results[sk] == mkInt(1));

  testSat(all(x, exists(y, mk(term::Eq, x, mk(term::Sub, mkInt(2), y)))), true);
  assert(skolems.n == 1);
  sk = skolems[0];
  assert(typeof(sk) == internType(type::Int, type::Int));
  assert(call(sk, 0) == 2);
  assert(call(sk, 1) == 1);
  assert(call(sk, 2) == 0);

  testSat(all(x, exists(y, mk(term::Eq, y, mk(term::Sub, mkInt(2), x)))), true);
  assert(skolems.n == 1);
  sk = skolems[0];
  assert(typeof(sk) == internType(type::Int, type::Int));
  assert(call(sk, 0) == 2);
  assert(call(sk, 1) == 1);
  assert(call(sk, 2) == 0);

  // this is a test of the interaction of equivalence with existential
  // qualifiers, which is nontrivial since existential qualifiers translate into
  // structurally different things depending on polarity, but it is not a very
  // tough test since the terms are too small to need renaming, which is where
  // it really gets difficult
  testSat(eqv(p2, exists(x, mk(term::Eq, x, mkInt(2)))), true);
  assert(results[p2] == term::True);
  assert(skolems.n == 1);
  sk = skolems[0];
  assert(typeof(sk) == type::Int);
  assert(results[sk] == mkInt(2));

  // test the expanded CNF code

  cnfx::testSat(term::True, true);
  cnfx::testSat(term::False, false);
  // duplicate is intentional
  cnfx::testSat(term::False, false);

  cnfx::testSat(intern(term::Eq, a, mkInt(1)), true);
  cnfx::testSat(intern(term::Eq, b, mkInt(2)), true);
  cnfx::testSat(intern(term::Eq, b, mkInt(intRange)), false);

  cnfx::testSat(intern(term::Not, term::False), true);
  cnfx::testSat(intern(term::Not, term::True), false);

  cnfx::testSat(intern(term::And, term::False, term::False, term::False),
                false);
  cnfx::testSat(intern(term::And, term::False, term::False, term::True), false);
  cnfx::testSat(intern(term::And, term::False, term::True, term::False), false);
  cnfx::testSat(intern(term::And, term::False, term::True, term::True), false);
  cnfx::testSat(intern(term::And, term::True, term::False, term::False), false);
  cnfx::testSat(intern(term::And, term::True, term::False, term::True), false);
  cnfx::testSat(intern(term::And, term::True, term::True, term::False), false);
  cnfx::testSat(intern(term::And, term::True, term::True, term::True), true);

  cnfx::testSat(intern(term::Or, term::False, term::False, term::False), false);
  cnfx::testSat(intern(term::Or, term::False, term::False, term::True), true);
  cnfx::testSat(intern(term::Or, term::False, term::True, term::False), true);
  cnfx::testSat(intern(term::Or, term::False, term::True, term::True), true);
  cnfx::testSat(intern(term::Or, term::True, term::False, term::False), true);
  cnfx::testSat(intern(term::Or, term::True, term::False, term::True), true);
  cnfx::testSat(intern(term::Or, term::True, term::True, term::False), true);
  cnfx::testSat(intern(term::Or, term::True, term::True, term::True), true);

  cnfx::testSat(intern(term::Eqv, term::False, term::True), false);
  cnfx::testSat(intern(term::Eqv, term::True, term::False), false);
  cnfx::testSat(intern(term::Eqv, term::False, term::False), true);
  cnfx::testSat(intern(term::Eqv, term::True, term::True), true);

  cnfx::testSat(p, true);
  cnfx::testSat(intern(term::Not, p), true);
  cnfx::testSat(intern(term::And, p, p), true);
  cnfx::testSat(intern(term::And, p, intern(term::Not, p)), false);
  cnfx::testSat(intern(term::And, p, intern(term::Not, q)), true);
  cnfx::testSat(intern(term::Or, p, p), true);
  cnfx::testSat(intern(term::Or, p, intern(term::Not, p)), true);
  cnfx::testSat(intern(term::Or, p, intern(term::Not, q)), true);
  cnfx::testSat(intern(term::Eqv, p, p), true);
  cnfx::testSat(intern(term::Eqv, p, q), true);
  cnfx::testSat(intern(term::Eqv, p, intern(term::Not, p)), false);
  cnfx::testSat(intern(term::Imp, p, p), true);
  cnfx::testSat(intern(term::Imp, p, intern(term::Not, p)), true);
  cnfx::testSat(intern(term::Imp, p, q), true);

  cnfx::testSat(intern(term::Eq, intern(term::Add, a, mkInt(1)), mkInt(1)),
                true);
  cnfx::testSat(intern(term::Eq, intern(term::Add, a, mkInt(1)), mkInt(2)),
                true);

  cnfx::testSat(eqv(p1, eqv(p2, eqv(p1, p2))), true);
  cnfx::testSat(eqv(p1, eqv(p2, eqv(p3, eqv(p1, eqv(p2, p3))))), true);
  cnfx::testSat(
      eqv(p1, eqv(p2, eqv(p3, eqv(p4, eqv(p1, eqv(p2, eqv(p3, p4))))))), true);
  cnfx::testSat(
      eqv(p1,
          eqv(p2,
              eqv(p3,
                  eqv(p4, eqv(p4, eqv(p5, eqv(p2, eqv(p3, eqv(p4, p5))))))))),
      true);

  cnfx::testSat(all(x, intern(term::Eq, x, mkInt(1))), false);
  cnfx::testSat(all(x, all(y, intern(term::Eq, x, x))), true);
  cnfx::testSat(all(x, all(y, intern(term::And, intern(term::Eq, x, x),
                                     intern(term::Eq, y, y)))),
                true);
  cnfx::testSat(all(x, all(y, intern(term::Eq, x, y))), false);
  cnfx::testSat(all(x, all(y, intern(term::And, intern(term::Eq, x, y),
                                     intern(term::Eq, y, x)))),
                false);
  cnfx::testSat(exists(x, intern(term::Eq, x, mkInt(1))), true);
  cnfx::testSat(
      exists(x, exists(y, intern(term::Eq, intern(term::Add, x, y), mkInt(1)))),
      true);
  cnfx::testSat(
      exists(x, exists(y, exists(z, intern(term::And, intern(term::Lt, x, y),
                                           intern(term::Lt, y, z))))),
      true);
  cnfx::testSat(exists(x, all(y, intern(term::Eq, x, y))), false);

  // searching for functions
  cnfx::testSat(all(x, intern(term::Eq, intern(term::Call, f1, x), mkInt(2))),
                true);

  cnfx::testSat(all(x, exists(y, intern(term::Eq, x, y))), true);

  cnfx::testSat(
      all(x, all(y, intern(term::Eq, intern(term::Call, f2, x, y), mkInt(2)))),
      true);

  cnfx::testSat(all(x, intern(term::Call, q1, x)), true);

  cnfx::testSat(all(x, all(y, intern(term::Call, q2, x, y))), true);

  // returning results
  // a=1
  cnfx::testSat(intern(term::Eq, a, mkInt(1)), true);
  assert(results[a] == mkInt(1));

  // q1(x)
  cnfx::testSat(all(x, intern(term::Call, q1, x)), true);
  assert(call(q1, 0) == 1);
  assert(call(q1, 1) == 1);
  assert(call(q1, 2) == 1);

  // f1(x)=2
  cnfx::testSat(all(x, intern(term::Eq, intern(term::Call, f1, x), mkInt(2))),
                true);
  assert(call(f1, 0) == 2);
  assert(call(f1, 1) == 2);
  assert(call(f1, 2) == 2);

  // q2(x,y)
  cnfx::testSat(all(x, all(y, intern(term::Call, q2, x, y))), true);
  assert(call(q2, 0, 0) == 1);
  assert(call(q2, 0, 1) == 1);
  assert(call(q2, 0, 2) == 1);
  assert(call(q2, 1, 0) == 1);
  assert(call(q2, 1, 1) == 1);
  assert(call(q2, 1, 2) == 1);
  assert(call(q2, 2, 0) == 1);
  assert(call(q2, 2, 1) == 1);
  assert(call(q2, 2, 2) == 1);

  // q1(x) <=> x=2
  cnfx::testSat(
      all(x, mk(term::Eqv, mk(term::Call, q1, x), mk(term::Eq, x, mkInt(2)))),
      true);
  assert(call(q1, 0) == 0);
  assert(call(q1, 1) == 0);
  assert(call(q1, 2) == 1);

  // q2(x,y) <=> x=y
  cnfx::testSat(all(x, all(y, mk(term::Eqv, mk(term::Call, q2, x, y),
                                 mk(term::Eq, x, y)))),
                true);
  assert(call(q2, 0, 0) == 1);
  assert(call(q2, 0, 1) == 0);
  assert(call(q2, 0, 2) == 0);
  assert(call(q2, 1, 0) == 0);
  assert(call(q2, 1, 1) == 1);
  assert(call(q2, 1, 2) == 0);
  assert(call(q2, 2, 0) == 0);
  assert(call(q2, 2, 1) == 0);
  assert(call(q2, 2, 2) == 1);

  cnfx::testSat(exists(x, intern(term::Eq, x, mkInt(1))), true);
  assert(skolems.n == 1);
  sk = skolems[0];
  assert(typeof(sk) == type::Int);
  assert(results[sk] == mkInt(1));

  cnfx::testSat(all(x, exists(y, mk(term::Eq, x, mk(term::Sub, mkInt(2), y)))),
                true);
  assert(skolems.n == 1);
  sk = skolems[0];
  assert(typeof(sk) == internType(type::Int, type::Int));
  assert(call(sk, 0) == 2);
  assert(call(sk, 1) == 1);
  assert(call(sk, 2) == 0);

  cnfx::testSat(all(x, exists(y, mk(term::Eq, y, mk(term::Sub, mkInt(2), x)))),
                true);
  assert(skolems.n == 1);
  sk = skolems[0];
  assert(typeof(sk) == internType(type::Int, type::Int));
  assert(call(sk, 0) == 2);
  assert(call(sk, 1) == 1);
  assert(call(sk, 2) == 0);

  // this is a test of the interaction of equivalence with existential
  // qualifiers, which is nontrivial since existential qualifiers translate into
  // structurally different things depending on polarity, but it is not a very
  // tough test since the terms are too small to need renaming, which is where
  // it really gets difficult
  cnfx::testSat(eqv(p2, exists(x, mk(term::Eq, x, mkInt(2)))), true);
  assert(results[p2] == term::True);
  assert(skolems.n == 1);
  sk = skolems[0];
  assert(typeof(sk) == type::Int);
  assert(results[sk] == mkInt(2));

  assert(eq(cnfx::distrib(mk(term::And, p1, mk(term::And, p2, p3))),
            mk(term::And, p1, p2, p3)));
  assert(eq(cnfx::distrib(mk(term::Or, p1, mk(term::And, p2, p3))),
            mk(term::And, mk(term::Or, p1, p2), mk(term::Or, p1, p3))));
  cnfx::testSat(mk(term::Eq, x, x), true);

  // a big term to trigger equivalence renaming during CNF conversion, while
  // being obviously a no-op so as not to interfere with the logic it is
  // attached to; of course it could be trivially evaluated and discarded on the
  // fly, but the CNF conversion does not bother looking for that, because such
  // trivial terms do not show up often in real problems
  auto big = eqv(term::True,
                 eqv(term::True, eqv(term::True, eqv(term::True, term::True))));
  cnfx::testSat(
      eqv(p2, mk(term::And, big, exists(x, mk(term::Eq, x, mkInt(2))))), true);
  assert(results[p2] == term::True);
  assert(skolems.n == 1);
  sk = skolems[0];
  assert(typeof(sk) == type::Int);
  assert(results[sk] == mkInt(2));
}
#endif
