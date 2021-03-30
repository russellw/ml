#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

// fast tests always run in debug build
#ifdef DEBUG
namespace {
vec<term> neg, pos;

bool eq(term a, term b) {
  vec<pair<term, term>> unified;
  return unify(a, b, unified) && !unified.n;
}

term fn(type t, sym *s) {
  if (s->t == type::none)
    s->t = t;
  assert(s->t == t);
  return tag(term::Sym, s);
}

clause *mkClause() {
  auto nn = neg.n;
  auto pn = pos.n;
  auto n = nn + pn;
  neg.n = pos.n = 0;
  memcpy(neg.p + nn, pos.p, pn * sizeof *pos.p);

  auto c = (clause *)xmalloc(offsetof(clause, v) + n * sizeof *neg.p);
  memset(c, 0, offsetof(clause, v));
  c->nn = nn;
  c->n = n;
  memcpy(c->v, neg.p, n * sizeof *neg.p);
  return c;
}

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
} // namespace

void test() {
  {
    vec<char, 1> v;
    assert(v.n == 0);

    v.push_back('a');
    assert(v.n == 1);
    assert(v.back() == 'a');

    v.push_back('b');
    assert(v.n == 2);
    assert(v.back() == 'b');

    v.push_back('c');
    assert(v.n == 3);
    assert(v.back() == 'c');
  }

  auto x = var(type::Int, 0);
  assert(vari(x) == 0);

  auto y = var(type::Int, 1);
  assert(vari(y) == 1);

  assert(x != y);

  // a simple clause, x!=y
  neg.n = pos.n = 0;
  neg.push_back(intern(term::Eq, x, y));
  auto c = intern(neg, pos, how::none);

  // duplicate returns null
  neg.n = pos.n = 0;
  neg.push_back(intern(term::Eq, x, y));
  auto d = intern(neg, pos, how::none);
  assert(!d);

  // the duplicate check distinguishes between negative and positive literals
  neg.n = pos.n = 0;
  pos.push_back(intern(term::Eq, x, y));
  d = intern(neg, pos, how::none);
  assert(c != d);

  // False
  initSolver();
  cnf(term::False, 0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  assert(!intern(neg, pos, how::none));

  // True
  initSolver();
  cnf(term::True, 0);
  assert(problem.n == 0);

  // !False
  initSolver();
  cnf(mk(term::Not, term::False), 0);
  assert(problem.n == 0);

  // !True
  initSolver();
  cnf(mk(term::Not, term::True), 0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  assert(!intern(neg, pos, how::none));

  // False & False
  initSolver();
  cnf(mk(term::And, term::False, term::False), 0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  assert(!intern(neg, pos, how::none));

  // False & True
  initSolver();
  cnf(mk(term::And, term::False, term::True), 0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  assert(!intern(neg, pos, how::none));

  // True & False
  initSolver();
  cnf(mk(term::And, term::True, term::False), 0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  assert(!intern(neg, pos, how::none));

  // True & True
  initSolver();
  cnf(mk(term::And, term::True, term::True), 0);
  assert(problem.n == 0);

  // False | False
  initSolver();
  cnf(mk(term::Or, term::False, term::False), 0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  assert(!intern(neg, pos, how::none));

  // False | True
  initSolver();
  cnf(mk(term::Or, term::False, term::True), 0);
  assert(problem.n == 0);

  // True | False
  initSolver();
  cnf(mk(term::Or, term::True, term::False), 0);
  assert(problem.n == 0);

  // True | True
  initSolver();
  cnf(mk(term::Or, term::True, term::True), 0);
  assert(problem.n == 0);

  // False => False
  initSolver();
  cnf(mk(term::Imp, term::False, term::False), 0);
  assert(problem.n == 0);

  // False => True
  initSolver();
  cnf(mk(term::Imp, term::False, term::True), 0);
  assert(problem.n == 0);

  // True => False
  initSolver();
  cnf(mk(term::Imp, term::True, term::False), 0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  assert(!intern(neg, pos, how::none));

  // True => True
  initSolver();
  cnf(mk(term::Imp, term::True, term::True), 0);
  assert(problem.n == 0);

  // p
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(p, 0);
    assert(problem.n == 1);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));
  }

  // !p
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::Not, p), 0);
    assert(problem.n == 1);

    neg.n = 0;
    neg.push_back(p);
    pos.n = 0;
    assert(!intern(neg, pos, how::none));
  }

  // p & p
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::And, p, p), 0);
    assert(problem.n == 1);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));
  }

  // p & !p
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::And, p, mk(term::Not, p)), 0);
    assert(problem.n == 2);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    neg.push_back(p);
    pos.n = 0;
    assert(!intern(neg, pos, how::none));
  }

  // p | p
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::Or, p, p), 0);
    assert(problem.n == 1);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));
  }

  // p | !p
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::Or, p, mk(term::Not, p)), 0);
    assert(problem.n == 0);
  }

  // !(p & p)
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::Not, mk(term::And, p, p)), 0);
    assert(problem.n == 1);

    neg.n = 0;
    neg.push_back(p);
    pos.n = 0;
    assert(!intern(neg, pos, how::none));
  }

  // !(p & !p)
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::Not, mk(term::And, p, mk(term::Not, p))), 0);
    assert(problem.n == 0);
  }

  // !(p | p)
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::Not, mk(term::Or, p, p)), 0);
    assert(problem.n == 1);

    neg.n = 0;
    neg.push_back(p);
    pos.n = 0;
    assert(!intern(neg, pos, how::none));
  }

  // !(p | !p)
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::Not, mk(term::Or, p, mk(term::Not, p))), 0);
    assert(problem.n == 2);

    neg.n = 0;
    neg.push_back(p);
    pos.n = 0;
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));
  }

  // p => p
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    cnf(mk(term::Imp, p, p), 0);
    assert(problem.n == 0);
  }

  // p & q
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    cnf(mk(term::And, p, q), 0);
    assert(problem.n == 2);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(q);
    assert(!intern(neg, pos, how::none));
  }

  // p & !q
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    cnf(mk(term::And, p, mk(term::Not, q)), 0);
    assert(problem.n == 2);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    neg.push_back(q);
    pos.n = 0;
    assert(!intern(neg, pos, how::none));
  }

  // p | q
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    cnf(mk(term::Or, p, q), 0);
    assert(problem.n == 1);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    pos.push_back(q);
    assert(!intern(neg, pos, how::none));
  }

  // p | !q
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    cnf(mk(term::Or, p, mk(term::Not, q)), 0);
    assert(problem.n == 1);

    neg.n = 0;
    neg.push_back(q);
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));
  }

  // p <=> q
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    cnf(mk(term::Eqv, p, q), 0);
    assert(problem.n == 2);

    neg.n = 0;
    neg.push_back(p);
    pos.n = 0;
    pos.push_back(q);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    neg.push_back(q);
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));
  }

  // p <~> q
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    cnf(mk(term::Not, mk(term::Eqv, p, q)), 0);
    assert(problem.n == 2);

    neg.n = 0;
    neg.push_back(p);
    neg.push_back(q);
    pos.n = 0;
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    pos.push_back(q);
    assert(!intern(neg, pos, how::none));
  }

  // p & q & r
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    auto r = fn(type::Bool, intern("r"));
    cnf(mk(term::And, p, q, r), 0);
    assert(problem.n == 3);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(q);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(r);
    assert(!intern(neg, pos, how::none));
  }

  // p & (q & r)
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    auto r = fn(type::Bool, intern("r"));
    cnf(mk(term::And, p, mk(term::And, q, r)), 0);
    assert(problem.n == 3);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(q);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(r);
    assert(!intern(neg, pos, how::none));
  }

  // (p & q) & r
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    auto r = fn(type::Bool, intern("r"));
    cnf(mk(term::And, mk(term::And, p, q), r), 0);
    assert(problem.n == 3);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(q);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(r);
    assert(!intern(neg, pos, how::none));
  }

  // p | q | r
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    auto r = fn(type::Bool, intern("r"));
    cnf(mk(term::Or, p, q, r), 0);
    assert(problem.n == 1);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    pos.push_back(q);
    pos.push_back(r);
    assert(!intern(neg, pos, how::none));
  }

  // p | (q | r)
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    auto r = fn(type::Bool, intern("r"));
    cnf(mk(term::Or, p, mk(term::Or, q, r)), 0);
    assert(problem.n == 1);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    pos.push_back(q);
    pos.push_back(r);
    assert(!intern(neg, pos, how::none));
  }

  // (p | q) | r
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    auto r = fn(type::Bool, intern("r"));
    cnf(mk(term::Or, mk(term::Or, p, q), r), 0);
    assert(problem.n == 1);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    pos.push_back(q);
    pos.push_back(r);
    assert(!intern(neg, pos, how::none));
  }

  // (p & q) | r
  {
    initSolver();
    auto p = fn(type::Bool, intern("p"));
    auto q = fn(type::Bool, intern("q"));
    auto r = fn(type::Bool, intern("r"));
    cnf(mk(term::Or, mk(term::And, p, q), r), 0);
    assert(problem.n == 2);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(p);
    pos.push_back(r);
    assert(!intern(neg, pos, how::none));

    neg.n = 0;
    pos.n = 0;
    pos.push_back(q);
    pos.push_back(r);
    assert(!intern(neg, pos, how::none));
  }

  // a = b
  {
    initSolver();
    auto a = fn(type::Int, intern("a"));
    auto b = fn(type::Int, intern("b"));
    cnf(intern(term::Eq, a, b), 0);
    assert(problem.n == 1);

    neg.n = 0;
    pos.n = 0;
    pos.push_back(intern(term::Eq, a, b));
    assert(!intern(neg, pos, how::none));
  }

  // 1 = 1
  initSolver();
  cnf(intern(term::Eq, mkInt(1), mkInt(1)), 0);
  assert(problem.n == 0);

  // 1 = 2
  initSolver();
  cnf(intern(term::Eq, mkInt(1), mkInt(2)), 0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  assert(!intern(neg, pos, how::none));

  // X=Y
  initSolver();
  cnf(intern(term::Eq, var(type::Individual, 0), var(type::Individual, 1)), 0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  pos.push_back(
      intern(term::Eq, var(type::Individual, 0), var(type::Individual, 1)));
  assert(!intern(neg, pos, how::none));

  // X=Y
  initSolver();
  cnf(intern(term::Eq, var(type::Individual, 10), var(type::Individual, 11)),
      0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  pos.push_back(
      intern(term::Eq, var(type::Individual, 0), var(type::Individual, 1)));
  assert(!intern(neg, pos, how::none));

  // ![X,Y]:X=Y
  initSolver();
  cnf(intern(term::All,
             intern(term::Eq, var(type::Individual, 10),
                    var(type::Individual, 11)),
             var(type::Individual, 10), var(type::Individual, 11)),
      0);
  assert(problem.n == 1);

  neg.n = 0;
  pos.n = 0;
  pos.push_back(
      intern(term::Eq, var(type::Individual, 0), var(type::Individual, 1)));
  assert(!intern(neg, pos, how::none));

  {
    auto red = fn(type::Bool, intern("red"));
    auto redp = (sym *)rest(red);
    assert(redp->t == type::Bool);

    auto green = fn(type::Bool, intern("green"));
    auto greenp = (sym *)rest(green);
    assert(greenp->t == type::Bool);

    auto blue = fn(type::Bool, intern("blue"));
    auto bluep = (sym *)rest(blue);
    assert(bluep->t == type::Bool);

    assert(redp == intern("red"));
    assert(greenp == intern("green"));
    assert(bluep == intern("blue"));
  }

  initSyms();
  auto a = fn(type::Individual, intern("a"));

  {
    vec<term> freeVars;
    getFreeVars(a, freeVars);
    assert(freeVars.n == 0);
  }

  {
    vec<term> freeVars;
    getFreeVars(x, freeVars);
    assert(freeVars.n == 1);
    assert(freeVars[0] == x);
  }

  {
    vec<term> freeVars;
    getFreeVars(intern(term::Eq, x, x), freeVars);
    assert(freeVars.n == 1);
    assert(freeVars[0] == x);
  }

  {
    vec<term> freeVars;
    getFreeVars(intern(term::Eq, x, y), freeVars);
    assert(freeVars.n == 2);
    assert(freeVars[0] == x);
    assert(freeVars[1] == y);
  }

  {
    vec<term> freeVars;
    getFreeVars(intern(term::Eq, a, a), freeVars);
    assert(freeVars.n == 0);
  }

  {
    vec<term> freeVars;
    getFreeVars(intern(term::All, intern(term::Eq, x, y), x), freeVars);
    assert(freeVars.n == 1);
    assert(freeVars[0] == y);
  }

  {
    Int x1;
    mpz_init_set_ui(x1.val, 1);
    auto a1 = tag(term::Int, intern(x1));
    auto y1 = (Int *)rest(a1);
    assert(!mpz_cmp_ui(y1->val, 1));

    Int x2;
    mpz_init_set_ui(x2.val, 2);
    auto a2 = tag(term::Int, intern(x2));
    auto y2 = (Int *)rest(a2);
    assert(!mpz_cmp_ui(y2->val, 2));

    Int x3;
    mpz_init(x3.val);
    mpz_add(x3.val, y1->val, y2->val);
    auto a3 = tag(term::Int, intern(x3));
    auto y3 = (Int *)rest(a3);
    assert(!mpz_cmp_ui(y3->val, 3));
  }

  {
    // subset of unify where only the first argument can be treated as a
    // variable to be matched against the second argument. applied to the same
    // test cases as unify, gives the same results in some cases, but different
    // results in others. in particular, has no notion of an occurs check; in
    // actual use, it is assumed that the arguments will have disjoint variables
    auto a = fn(type::Individual, intern("a"));
    auto b = fn(type::Individual, intern("b"));
    auto f1 = fn(internType(type::Individual, type::Individual), intern("f1"));
    auto f2 =
        fn(internType(type::Individual, type::Individual, type::Individual),
           intern("f2"));
    auto g1 = fn(internType(type::Individual, type::Individual), intern("g1"));
    auto x = var(type::Individual, 0);
    auto y = var(type::Individual, 1);
    auto z = var(type::Individual, 2);
    vec<pair<term, term>> matched;

    // Succeeds. (tautology)
    matched.n = 0;
    assert(match(a, a, matched));
    assert(matched.n == 0);

    // a and b do not match
    matched.n = 0;
    assert(!match(a, b, matched));

    // Succeeds. (tautology)
    matched.n = 0;
    assert(match(x, x, matched));
    assert(matched.n == 0);

    // x is not matched with the constant a, because the variable is on the
    // right-hand side
    matched.n = 0;
    assert(!match(a, x, matched));

    // x and y are aliased
    matched.n = 0;
    assert(match(x, y, matched));
    assert(matched.n == 1);
    assert(replace(x, matched) == replace(y, matched));

    // Function and constant symbols match, x is unified with the constant b
    matched.n = 0;
    assert(match(intern(term::Call, f2, a, x), intern(term::Call, f2, a, b),
                 matched));
    assert(matched.n == 1);
    assert(replace(x, matched) == b);

    // f and g do not match
    matched.n = 0;
    assert(
        !match(intern(term::Call, f1, a), intern(term::Call, g1, a), matched));

    // x and y are aliased
    matched.n = 0;
    assert(
        match(intern(term::Call, f1, x), intern(term::Call, f1, y), matched));
    assert(matched.n == 1);
    assert(replace(x, matched) == replace(y, matched));

    // f and g do not match
    matched.n = 0;
    assert(
        !match(intern(term::Call, f1, x), intern(term::Call, g1, y), matched));

    // Fails. The f function symbols have different arity
    matched.n = 0;
    assert(!match(intern(term::Call, f1, x), intern(term::Call, f2, y, z),
                  matched));

    // Does not match y with the term g1(x), because the variable is on the
    // right-hand side
    matched.n = 0;
    assert(!match(intern(term::Call, f1, intern(term::Call, g1, x)),
                  intern(term::Call, f1, y), matched));

    // Does not match, because the variable is on the right-hand side
    matched.n = 0;
    assert(!match(intern(term::Call, f2, intern(term::Call, g1, x), x),
                  intern(term::Call, f2, y, a), matched));

    // Returns false in first-order logic and many modern Prolog dialects
    // (enforced by the occurs check) but returns true here because match has no
    // notion of an occurs check
    matched.n = 0;
    assert(match(x, intern(term::Call, f1, x), matched));
    assert(matched.n == 1);

    // Both x and y are unified with the constant a
    matched.n = 0;
    assert(match(x, y, matched));
    assert(match(y, a, matched));
    assert(matched.n == 2);
    assert(replace(x, matched) == a);
    assert(replace(y, matched) == a);

    // Fails this time, because the variable is on the right-hand side
    matched.n = 0;
    assert(!match(a, y, matched));

    // Fails. a and b do not match, so x can't be unified with both
    matched.n = 0;
    assert(match(x, a, matched));
    assert(!match(b, x, matched));
  }

  {
    Rat x1;
    mpq_init(x1.val);
    mpq_set_ui(x1.val, 1, 1);
    auto a1 = tag(term::Rat, intern(x1));
    auto y1 = (Rat *)rest(a1);
    assert(!mpq_cmp_ui(y1->val, 1, 1));

    Rat x2;
    mpq_init(x2.val);
    mpq_set_ui(x2.val, 2, 1);
    auto a2 = tag(term::Rat, intern(x2));
    auto y2 = (Rat *)rest(a2);
    assert(!mpq_cmp_ui(y2->val, 2, 1));

    Rat x3;
    mpq_init(x3.val);
    mpq_add(x3.val, y1->val, y2->val);
    auto a3 = tag(term::Rat, intern(x3));
    auto y3 = (Rat *)rest(a3);
    assert(!mpq_cmp_ui(y3->val, 3, 1));
  }

  {
    initSyms();
    auto a = fn(type::Int, intern("a"));
    auto b = fn(type::Int, intern("b"));

    assert(simplify(intern(term::Eq, a, a)) == term::True);
    assert(simplify(intern(term::Eq, a, b)) == intern(term::Eq, a, b));
    assert(simplify(intern(term::Eq, mkInt(1), mkInt(1))) == term::True);
    assert(simplify(intern(term::Eq, mkInt(1), mkInt(2))) == term::False);
    assert(simplify(intern(term::Eq, mkRat(1, 3), mkRat(2, 6))) == term::True);

    assert(simplify(intern(term::Lt, a, a)) == term::False);
    assert(simplify(intern(term::Lt, a, b)) == intern(term::Lt, a, b));
    assert(simplify(intern(term::Lt, mkInt(1), mkInt(2))) == term::True);
    assert(simplify(intern(term::Lt, mkInt(2), mkInt(1))) == term::False);
    assert(simplify(intern(term::Lt, mkRat(1, 3), mkRat(1, 2))) == term::True);
    assert(simplify(intern(term::Lt, a, mkInt(1))) ==
           intern(term::Lt, a, mkInt(1)));

    assert(simplify(intern(term::Le, a, a)) == term::True);
    assert(simplify(intern(term::Le, a, b)) == intern(term::Le, a, b));
    assert(simplify(intern(term::Le, mkInt(1), mkInt(2))) == term::True);
    assert(simplify(intern(term::Le, mkInt(2), mkInt(1))) == term::False);
    assert(simplify(intern(term::Le, mkRat(1, 3), mkRat(1, 2))) == term::True);
    assert(simplify(intern(term::Le, mkInt(1), b)) ==
           intern(term::Le, mkInt(1), b));

    assert(simplify(intern(term::Add, a, b)) == intern(term::Add, a, b));
    assert(simplify(intern(term::Add, mkInt(1), mkInt(2))) == mkInt(3));
    assert(simplify(intern(term::Add, mkRat(1, 3), mkRat(1, 2))) ==
           mkRat(5, 6));

    assert(simplify(intern(term::Sub, a, mkInt(1))) ==
           intern(term::Sub, a, mkInt(1)));
    assert(simplify(intern(term::Sub, mkInt(10), mkInt(2))) == mkInt(8));
    assert(simplify(intern(term::Sub, mkRat(1, 2), mkRat(1, 3))) ==
           mkRat(1, 6));

    assert(simplify(intern(term::Mul, mkInt(1), b)) ==
           intern(term::Mul, mkInt(1), b));
    assert(simplify(intern(term::Mul, mkInt(10), mkInt(2))) == mkInt(20));
    assert(simplify(intern(term::Mul, mkRat(1, 2), mkRat(1, 7))) ==
           mkRat(1, 14));

    assert(simplify(intern(term::Div, mkRat(1, 2), mkRat(1, 7))) ==
           mkRat(7, 2));

    assert(simplify(intern(term::Minus, mkInt(1))) == mkInt(-1));
    assert(simplify(intern(term::Minus, mkReal(-1.5))) == mkReal(1.5));

    assert(simplify(intern(term::DivF, mkInt(5), mkInt(3))) == mkInt(1));
    assert(simplify(intern(term::DivF, mkInt(-5), mkInt(3))) == mkInt(-2));
    assert(simplify(intern(term::DivF, mkInt(5), mkInt(-3))) == mkInt(-2));
    assert(simplify(intern(term::DivF, mkInt(-5), mkInt(-3))) == mkInt(1));
    assert(simplify(intern(term::DivF, mkRat(5), mkRat(3))) == mkRat(1));
    assert(simplify(intern(term::DivF, mkRat(-5), mkRat(3))) == mkRat(-2));
    assert(simplify(intern(term::DivF, mkRat(5), mkRat(-3))) == mkRat(-2));
    assert(simplify(intern(term::DivF, mkRat(-5), mkRat(-3))) == mkRat(1));

    assert(simplify(intern(term::RemF, mkInt(5), mkInt(3))) == mkInt(2));
    assert(simplify(intern(term::RemF, mkInt(-5), mkInt(3))) == mkInt(1));
    assert(simplify(intern(term::RemF, mkInt(5), mkInt(-3))) == mkInt(-1));
    assert(simplify(intern(term::RemF, mkInt(-5), mkInt(-3))) == mkInt(-2));
    assert(simplify(intern(term::RemF, mkRat(5), mkRat(3))) == mkRat(2));
    assert(simplify(intern(term::RemF, mkRat(-5), mkRat(3))) == mkRat(1));
    assert(simplify(intern(term::RemF, mkRat(5), mkRat(-3))) == mkRat(-1));
    assert(simplify(intern(term::RemF, mkRat(-5), mkRat(-3))) == mkRat(-2));

    assert(simplify(intern(term::DivT, mkInt(5), mkInt(3))) == mkInt(5 / 3));
    assert(simplify(intern(term::DivT, mkInt(-5), mkInt(3))) == mkInt(-5 / 3));
    assert(simplify(intern(term::DivT, mkInt(5), mkInt(-3))) == mkInt(5 / -3));
    assert(simplify(intern(term::DivT, mkInt(-5), mkInt(-3))) ==
           mkInt(-5 / -3));
    assert(simplify(intern(term::DivT, mkInt(5), mkInt(3))) == mkInt(1));
    assert(simplify(intern(term::DivT, mkInt(-5), mkInt(3))) == mkInt(-1));
    assert(simplify(intern(term::DivT, mkInt(5), mkInt(-3))) == mkInt(-1));
    assert(simplify(intern(term::DivT, mkInt(-5), mkInt(-3))) == mkInt(1));
    assert(simplify(intern(term::DivT, mkRat(5), mkRat(3))) == mkRat(1));
    assert(simplify(intern(term::DivT, mkRat(-5), mkRat(3))) == mkRat(-1));
    assert(simplify(intern(term::DivT, mkRat(5), mkRat(-3))) == mkRat(-1));
    assert(simplify(intern(term::DivT, mkRat(-5), mkRat(-3))) == mkRat(1));

    assert(simplify(intern(term::RemT, mkInt(5), mkInt(3))) == mkInt(5 % 3));
    assert(simplify(intern(term::RemT, mkInt(-5), mkInt(3))) == mkInt(-5 % 3));
    assert(simplify(intern(term::RemT, mkInt(5), mkInt(-3))) == mkInt(5 % -3));
    assert(simplify(intern(term::RemT, mkInt(-5), mkInt(-3))) ==
           mkInt(-5 % -3));
    assert(simplify(intern(term::RemT, mkInt(5), mkInt(3))) == mkInt(2));
    assert(simplify(intern(term::RemT, mkInt(-5), mkInt(3))) == mkInt(-2));
    assert(simplify(intern(term::RemT, mkInt(5), mkInt(-3))) == mkInt(2));
    assert(simplify(intern(term::RemT, mkInt(-5), mkInt(-3))) == mkInt(-2));
    assert(simplify(intern(term::RemT, mkRat(5), mkRat(3))) == mkRat(2));
    assert(simplify(intern(term::RemT, mkRat(-5), mkRat(3))) == mkRat(-2));
    assert(simplify(intern(term::RemT, mkRat(5), mkRat(-3))) == mkRat(2));
    assert(simplify(intern(term::RemT, mkRat(-5), mkRat(-3))) == mkRat(-2));

    assert(simplify(intern(term::DivE, mkInt(7), mkInt(3))) == mkInt(2));
    assert(simplify(intern(term::DivE, mkInt(-7), mkInt(3))) == mkInt(-3));
    assert(simplify(intern(term::DivE, mkInt(7), mkInt(-3))) == mkInt(-2));
    assert(simplify(intern(term::DivE, mkInt(-7), mkInt(-3))) == mkInt(3));
    assert(simplify(intern(term::DivE, mkRat(7), mkRat(3))) == mkRat(2));
    assert(simplify(intern(term::DivE, mkRat(-7), mkRat(3))) == mkRat(-3));
    assert(simplify(intern(term::DivE, mkRat(7), mkRat(-3))) == mkRat(-2));
    assert(simplify(intern(term::DivE, mkRat(-7), mkRat(-3))) == mkRat(3));

    assert(simplify(intern(term::RemE, mkInt(7), mkInt(3))) == mkInt(1));
    assert(simplify(intern(term::RemE, mkInt(-7), mkInt(3))) == mkInt(2));
    assert(simplify(intern(term::RemE, mkInt(7), mkInt(-3))) == mkInt(1));
    assert(simplify(intern(term::RemE, mkInt(-7), mkInt(-3))) == mkInt(2));
    assert(simplify(intern(term::RemE, mkRat(7), mkRat(3))) == mkRat(1));
    assert(simplify(intern(term::RemE, mkRat(-7), mkRat(3))) == mkRat(2));
    assert(simplify(intern(term::RemE, mkRat(7), mkRat(-3))) == mkRat(1));
    assert(simplify(intern(term::RemE, mkRat(-7), mkRat(-3))) == mkRat(2));

    assert(simplify(intern(term::Ceil, mkInt(0))) == mkInt(0));
    assert(simplify(intern(term::Ceil, mkRat(0))) == mkRat(0));
    assert(simplify(intern(term::Ceil, mkRat(1, 10))) == mkRat(1));
    assert(simplify(intern(term::Ceil, mkRat(5, 10))) == mkRat(1));
    assert(simplify(intern(term::Ceil, mkRat(9, 10))) == mkRat(1));
    assert(simplify(intern(term::Ceil, mkRat(-1, 10))) == mkRat(0));
    assert(simplify(intern(term::Ceil, mkRat(-5, 10))) == mkRat(0));
    assert(simplify(intern(term::Ceil, mkRat(-9, 10))) == mkRat(0));

    assert(simplify(intern(term::Floor, mkInt(0))) == mkInt(0));
    assert(simplify(intern(term::Floor, mkRat(0))) == mkRat(0));
    assert(simplify(intern(term::Floor, mkRat(1, 10))) == mkRat(0));
    assert(simplify(intern(term::Floor, mkRat(5, 10))) == mkRat(0));
    assert(simplify(intern(term::Floor, mkRat(9, 10))) == mkRat(0));
    assert(simplify(intern(term::Floor, mkRat(-1, 10))) == mkRat(-1));
    assert(simplify(intern(term::Floor, mkRat(-5, 10))) == mkRat(-1));
    assert(simplify(intern(term::Floor, mkRat(-9, 10))) == mkRat(-1));

    assert(simplify(intern(term::Trunc, mkInt(0))) == mkInt(0));
    assert(simplify(intern(term::Trunc, mkRat(0))) == mkRat(0));
    assert(simplify(intern(term::Trunc, mkRat(1, 10))) == mkRat(0));
    assert(simplify(intern(term::Trunc, mkRat(5, 10))) == mkRat(0));
    assert(simplify(intern(term::Trunc, mkRat(9, 10))) == mkRat(0));
    assert(simplify(intern(term::Trunc, mkRat(-1, 10))) == mkRat(0));
    assert(simplify(intern(term::Trunc, mkRat(-5, 10))) == mkRat(0));
    assert(simplify(intern(term::Trunc, mkRat(-9, 10))) == mkRat(0));

    assert(simplify(intern(term::Round, mkInt(0))) == mkInt(0));
    assert(simplify(intern(term::Round, mkRat(0))) == mkRat(0));
    assert(simplify(intern(term::Round, mkRat(1, 10))) == mkRat(0));
    assert(simplify(intern(term::Round, mkRat(5, 10))) == mkRat(0));
    assert(simplify(intern(term::Round, mkRat(9, 10))) == mkRat(1));
    assert(simplify(intern(term::Round, mkRat(-1, 10))) == mkRat(0));
    assert(simplify(intern(term::Round, mkRat(-5, 10))) == mkRat(0));
    assert(simplify(intern(term::Round, mkRat(-9, 10))) == mkRat(-1));
    assert(simplify(intern(term::Round, mkRat(15, 10))) == mkRat(2));
    assert(simplify(intern(term::Round, mkRat(25, 10))) == mkRat(2));
    assert(simplify(intern(term::Round, mkRat(35, 10))) == mkRat(4));
    assert(simplify(intern(term::Round, mkRat(45, 10))) == mkRat(4));

    assert(simplify(intern(term::IsInt, a)) == term::True);
    assert(simplify(intern(term::IsInt, mkRat(5, 5))) == term::True);
    assert(simplify(intern(term::IsInt, mkRat(5, 10))) == term::False);

    assert(simplify(intern(term::IsRat, a)) == term::True);
    assert(simplify(intern(term::IsRat, mkRat(45, 10))) == term::True);
    assert(simplify(intern(term::IsRat, mkReal(2.5))) == term::True);

    assert(simplify(intern(term::ToInt, a)) == a);
    assert(simplify(intern(term::ToInt, mkInt(0))) == mkInt(0));
    assert(simplify(intern(term::ToInt, mkRat(0))) == mkInt(0));
    assert(simplify(intern(term::ToInt, mkRat(1, 10))) == mkInt(0));
    assert(simplify(intern(term::ToInt, mkRat(5, 10))) == mkInt(0));
    assert(simplify(intern(term::ToInt, mkRat(9, 10))) == mkInt(0));
    assert(simplify(intern(term::ToInt, mkRat(-1, 10))) == mkInt(-1));
    assert(simplify(intern(term::ToInt, mkRat(-5, 10))) == mkInt(-1));
    assert(simplify(intern(term::ToInt, mkRat(-9, 10))) == mkInt(-1));

    assert(simplify(intern(term::ToRat, a)) == intern(term::ToRat, a));
    assert(simplify(intern(term::ToRat, mkInt(7))) == mkRat(7));
    assert(simplify(intern(term::ToRat, mkRat(7))) == mkRat(7));
    assert(simplify(intern(term::ToRat, mkReal(7.0))) == mkRat(7));

    assert(simplify(intern(term::ToReal, a)) == intern(term::ToReal, a));
    assert(simplify(intern(term::ToReal, mkInt(7))) == mkReal(7));
    assert(simplify(intern(term::ToReal, mkRat(7))) == mkReal(7));
    assert(simplify(intern(term::ToReal, mkReal(7.0))) == mkReal(7));
  }

  {
    initSyms();
    auto a = fn(type::Individual, intern("a"));
    auto a1 = fn(internType(type::Individual, type::Individual), intern("a1"));
    auto b = fn(type::Individual, intern("b"));
    auto p = fn(type::Bool, intern("p"));
    auto p1 = fn(internType(type::Bool, type::Individual), intern("p1"));
    auto p2 = fn(internType(type::Bool, type::Individual, type::Individual),
                 intern("p2"));
    auto q = fn(type::Bool, intern("q"));
    auto q1 = fn(internType(type::Bool, type::Individual), intern("q1"));
    auto x = var(type::Individual, 0);
    auto y = var(type::Individual, 1);
    clause *c;
    clause *d;

    // false <= false
    c = mkClause();
    d = c;
    assert(subsumes(c, d));

    // false <= p
    c = mkClause();
    pos.push_back(p);
    d = mkClause();
    assert(subsumes(c, d));
    assert(!subsumes(d, c));

    // p <= p
    pos.push_back(p);
    c = mkClause();
    d = c;
    assert(subsumes(c, d));

    // !p <= !p
    neg.push_back(p);
    c = mkClause();
    d = c;
    assert(subsumes(c, d));

    // p <= p | p
    pos.push_back(p);
    c = mkClause();
    pos.push_back(p);
    pos.push_back(p);
    d = mkClause();
    assert(subsumes(c, d));
    assert(!subsumes(d, c));

    // p !<= !p
    pos.push_back(p);
    c = mkClause();
    neg.push_back(p);
    d = mkClause();
    assert(!subsumes(c, d));
    assert(!subsumes(d, c));

    // p | q <= q | p
    pos.push_back(p);
    pos.push_back(q);
    c = mkClause();
    pos.push_back(q);
    pos.push_back(p);
    d = mkClause();
    assert(subsumes(c, d));
    assert(subsumes(d, c));

    // p | q <= p | q | p
    pos.push_back(p);
    pos.push_back(q);
    c = mkClause();
    pos.push_back(p);
    pos.push_back(q);
    pos.push_back(p);
    d = mkClause();
    assert(subsumes(c, d));
    assert(!subsumes(d, c));

    // p(a) | p(b) | q(a) | q(b) | <= p(a) | q(a) | p(b) | q(b)
    pos.push_back(intern(term::Call, p1, a));
    pos.push_back(intern(term::Call, p1, b));
    pos.push_back(intern(term::Call, q1, a));
    pos.push_back(intern(term::Call, q1, b));
    c = mkClause();
    pos.push_back(intern(term::Call, p1, a));
    pos.push_back(intern(term::Call, q1, a));
    pos.push_back(intern(term::Call, p1, b));
    pos.push_back(intern(term::Call, q1, b));
    d = mkClause();
    assert(subsumes(c, d));
    assert(subsumes(d, c));

    // p(x,y) <= p(a,b)
    pos.push_back(intern(term::Call, p2, x, y));
    c = mkClause();
    pos.push_back(intern(term::Call, p2, a, b));
    d = mkClause();
    assert(subsumes(c, d));
    assert(!subsumes(d, c));

    // p(x,x) !<= p(a,b)
    pos.push_back(intern(term::Call, p2, x, x));
    c = mkClause();
    pos.push_back(intern(term::Call, p2, a, b));
    d = mkClause();
    assert(!subsumes(c, d));
    assert(!subsumes(d, c));

    // p(x) <= p(y)
    pos.push_back(intern(term::Call, p1, x));
    c = mkClause();
    pos.push_back(intern(term::Call, p1, y));
    d = mkClause();
    assert(subsumes(c, d));
    assert(subsumes(d, c));

    // p(x) | p(a(x)) | p(a(a(x))) <= p(y) | p(a(y)) | p(a(a(y)))
    pos.push_back(intern(term::Call, p1, x));
    pos.push_back(intern(term::Call, p1, intern(term::Call, a1, x)));
    pos.push_back(intern(term::Call, p1,
                         intern(term::Call, a1, intern(term::Call, a1, x))));
    c = mkClause();
    pos.push_back(intern(term::Call, p1, y));
    pos.push_back(intern(term::Call, p1, intern(term::Call, a1, y)));
    pos.push_back(intern(term::Call, p1,
                         intern(term::Call, a1, intern(term::Call, a1, y))));
    d = mkClause();
    assert(subsumes(c, d));
    assert(subsumes(d, c));

    // p(x) | p(a) <= p(a) | p(b)
    pos.push_back(intern(term::Call, p1, x));
    pos.push_back(intern(term::Call, p1, a));
    c = mkClause();
    pos.push_back(intern(term::Call, p1, a));
    pos.push_back(intern(term::Call, p1, b));
    d = mkClause();
    assert(subsumes(c, d));
    assert(!subsumes(d, c));

    // p(x) | p(a(x)) <= p(a(y)) | p(y)
    pos.push_back(intern(term::Call, p1, x));
    pos.push_back(intern(term::Call, p1, intern(term::Call, a1, x)));
    c = mkClause();
    pos.push_back(intern(term::Call, p1, intern(term::Call, a1, y)));
    pos.push_back(intern(term::Call, p1, y));
    d = mkClause();
    assert(subsumes(c, d));
    assert(subsumes(d, c));

    // p(x) | p(a(x)) | p(a(a(x))) <= p(a(a(y))) | p(a(y)) | p(y)
    pos.push_back(intern(term::Call, p1, x));
    pos.push_back(intern(term::Call, p1, intern(term::Call, a1, x)));
    pos.push_back(intern(term::Call, p1,
                         intern(term::Call, a1, intern(term::Call, a1, x))));
    c = mkClause();
    pos.push_back(intern(term::Call, p1,
                         intern(term::Call, a1, intern(term::Call, a1, y))));
    pos.push_back(intern(term::Call, p1, intern(term::Call, a1, y)));
    pos.push_back(intern(term::Call, p1, y));
    d = mkClause();
    assert(subsumes(c, d));
    assert(subsumes(d, c));

    // (a = x) <= (a = b)
    pos.push_back(intern(term::Eq, a, x));
    c = mkClause();
    pos.push_back(intern(term::Eq, a, b));
    d = mkClause();
    assert(subsumes(c, d));
    assert(!subsumes(d, c));

    // (x = a) <= (a = b)
    pos.push_back(intern(term::Eq, x, a));
    c = mkClause();
    pos.push_back(intern(term::Eq, a, b));
    d = mkClause();
    assert(subsumes(c, d));
    assert(!subsumes(d, c));

    // !p(y) | !p(x) | q(x) <= !p(a) | !p(b) | q(b)
    neg.push_back(intern(term::Call, p1, y));
    neg.push_back(intern(term::Call, p1, x));
    pos.push_back(intern(term::Call, q1, x));
    c = mkClause();
    neg.push_back(intern(term::Call, p1, a));
    neg.push_back(intern(term::Call, p1, b));
    pos.push_back(intern(term::Call, q1, b));
    d = mkClause();
    assert(subsumes(c, d));
    assert(!subsumes(d, c));

    // !p(x) | !p(y) | q(x) <= !p(a) | !p(b) | q(b)
    neg.push_back(intern(term::Call, p1, x));
    neg.push_back(intern(term::Call, p1, y));
    pos.push_back(intern(term::Call, q1, x));
    c = mkClause();
    neg.push_back(intern(term::Call, p1, a));
    neg.push_back(intern(term::Call, p1, b));
    pos.push_back(intern(term::Call, q1, b));
    d = mkClause();
    assert(subsumes(c, d));
    assert(!subsumes(d, c));

    // p(x,a(x)) !<= p(a(y),a(y))
    pos.push_back(intern(term::Call, p2, x, intern(term::Call, a1, x)));
    c = mkClause();
    pos.push_back(intern(term::Call, p2, intern(term::Call, a1, y),
                         intern(term::Call, a1, y)));
    d = mkClause();
    assert(!subsumes(c, d));
    assert(!subsumes(d, c));
  }

  assert(keyword(intern("cnf")) == k_cnf);
  assert(keyword(intern("cnf....", 3)) == k_cnf);
  assert(keyword(intern("fof")) == k_fof);
  assert(keyword(intern("tff")) == k_tff);
  assert(intern("") == intern("", 0));
  assert(intern("xyz") == intern("xyz", 3));

  {
    auto red = fn(type::Bool, intern("red"));
    auto green = fn(type::Bool, intern("green"));
    auto blue = fn(type::Bool, intern("blue"));

    auto a = intern(term::Not, red);
    assert(a == intern(term::Not, red));

    a = intern(term::And, red, green);
    assert(a == intern(term::And, red, green));

    vec<term> v;
    v.push_back(red);
    v.push_back(green);
    v.push_back(blue);
    a = intern(term::And, v);
    assert(a == intern(term::And, v));
  }

  {
    assert(typeof(var(type::Int, 13)) == type::Int);

    Int i1;
    mpz_init_set_ui(i1.val, 1);
    assert(typeof(tag(term::Int, intern(i1))) == type::Int);

    Rat r1;
    mpq_init(r1.val);
    mpq_set_ui(r1.val, 1, 3);
    assert(typeof(tag(term::Rat, intern(r1))) == type::Rat);
    mpq_init(r1.val);
    mpq_set_ui(r1.val, 1, 3);
    assert(typeof(tag(term::Real, intern(r1))) == type::Real);

    auto red = fn(type::Bool, intern("red"));
    assert(typeof(red) == type::Bool);
  }

  {
    auto bird = internType(intern("bird"));
    assert(bird == internType(intern("bird")));
    assert(!isCompound(bird));

    auto plane = internType(intern("plane"));
    assert(plane == internType(intern("plane")));
    assert(!isCompound(plane));

    assert(bird != plane);

    vec<type> v;
    v.push_back(type::Bool);
    v.push_back(type::Int);
    v.push_back(type::Int);
    auto t_predicate_int_int = internType(v);
    assert(t_predicate_int_int == internType(v));
    assert(isCompound(t_predicate_int_int));
    auto t = tcompoundp(t_predicate_int_int);
    assert(t->n == 3);
    assert(t->v[0] == type::Bool);
    assert(t->v[1] == type::Int);
    assert(t->v[2] == type::Int);

    v.n = 0;
    v.push_back(type::Bool);
    v.push_back(type::Rat);
    v.push_back(type::Rat);
    auto t_predicate_rat_rat = internType(v);
    assert(t_predicate_rat_rat == internType(v));
    assert(isCompound(t_predicate_rat_rat));
    t = tcompoundp(t_predicate_rat_rat);
    assert(t->n == 3);
    assert(t->v[0] == type::Bool);
    assert(t->v[1] == type::Rat);
    assert(t->v[2] == type::Rat);
  }

  {
    // https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
    initSyms();
    auto a = fn(type::Individual, intern("a"));
    auto b = fn(type::Individual, intern("b"));
    auto f1 = fn(internType(type::Individual, type::Individual), intern("f1"));
    auto f2 =
        fn(internType(type::Individual, type::Individual, type::Individual),
           intern("f2"));
    auto g1 = fn(internType(type::Individual, type::Individual), intern("g1"));
    auto x = var(type::Individual, 0);
    auto y = var(type::Individual, 1);
    auto z = var(type::Individual, 2);
    vec<pair<term, term>> unified;

    // Succeeds. (tautology)
    unified.n = 0;
    assert(unify(a, a, unified));
    assert(unified.n == 0);

    // a and b do not match
    unified.n = 0;
    assert(!unify(a, b, unified));

    // Succeeds. (tautology)
    unified.n = 0;
    assert(unify(x, x, unified));
    assert(unified.n == 0);

    // x is unified with the constant a
    unified.n = 0;
    assert(unify(a, x, unified));
    assert(unified.n == 1);
    assert(replace(x, unified) == a);

    // x and y are aliased
    unified.n = 0;
    assert(unify(x, y, unified));
    assert(unified.n == 1);
    assert(replace(x, unified) == replace(y, unified));

    // Function and constant symbols match, x is unified with the constant b
    unified.n = 0;
    assert(unify(intern(term::Call, f2, a, x), intern(term::Call, f2, a, b),
                 unified));
    assert(unified.n == 1);
    assert(replace(x, unified) == b);

    // f and g do not match
    unified.n = 0;
    assert(
        !unify(intern(term::Call, f1, a), intern(term::Call, g1, a), unified));

    // x and y are aliased
    unified.n = 0;
    assert(
        unify(intern(term::Call, f1, x), intern(term::Call, f1, y), unified));
    assert(unified.n == 1);
    assert(replace(x, unified) == replace(y, unified));

    // f and g do not match
    unified.n = 0;
    assert(
        !unify(intern(term::Call, f1, x), intern(term::Call, g1, y), unified));

    // Fails. The f function symbols have different arity
    unified.n = 0;
    assert(!unify(intern(term::Call, f1, x), intern(term::Call, f2, y, z),
                  unified));

    // Unifies y with the term g1(x)
    unified.n = 0;
    assert(unify(intern(term::Call, f1, intern(term::Call, g1, x)),
                 intern(term::Call, f1, y), unified));
    assert(unified.n == 1);
    assert(eq(replace(y, unified), intern(term::Call, g1, x)));

    // Unifies x with constant a, and y with the term g1(a)
    unified.n = 0;
    assert(unify(intern(term::Call, f2, intern(term::Call, g1, x), x),
                 intern(term::Call, f2, y, a), unified));
    assert(unified.n == 2);
    assert(replace(x, unified) == a);
    assert(eq(replace(y, unified), intern(term::Call, g1, a)));

    // Returns false in first-order logic and many modern Prolog dialects
    // (enforced by the occurs check).
    unified.n = 0;
    assert(!unify(x, intern(term::Call, f1, x), unified));

    // Both x and y are unified with the constant a
    unified.n = 0;
    assert(unify(x, y, unified));
    assert(unify(y, a, unified));
    assert(unified.n == 2);
    assert(replace(x, unified) == a);
    assert(replace(y, unified) == a);

    // As above (order of equations in set doesn't matter)
    unified.n = 0;
    assert(unify(a, y, unified));
    assert(unify(x, y, unified));
    assert(unified.n == 2);
    assert(replace(x, unified) == a);
    assert(replace(y, unified) == a);

    // Fails. a and b do not match, so x can't be unified with both
    unified.n = 0;
    assert(unify(x, a, unified));
    assert(!unify(b, x, unified));
  }

  initSolver();
  auto addInt =
      fn(internType(type::Int, type::Int, type::Int), intern("addInt"));

  // make sure unify correctly rejects terms with different types
  vec<pair<term, term>> unified;
  assert(unify(mk(term::Add, mkInt(1), mkInt(2)), var(type::Int, 0), unified));

  unified.n = 0;
  assert(!unify(mk(term::Add, mkInt(1), mkInt(2)), var(type::Individual, 0),
                unified));

  unified.n = 0;
  assert(unify(mk(term::Eq, mk(term::Add, mkInt(1), mkInt(2)),
                  mk(term::Add, mkInt(1), mkInt(2))),
               mk(term::Eq, var(type::Int, 0), var(type::Int, 1)), unified));

  unified.n = 0;
  assert(!unify(mk(term::Eq, mk(term::Add, mkInt(1), mkInt(2)),
                   mk(term::Add, mkInt(1), mkInt(2))),
                mk(term::Eq, var(type::Real, 0), var(type::Real, 1)), unified));

  unified.n = 0;
  assert(unify(intern(term::Eq, intern(term::Add, mkInt(1), mkInt(2)),
                      intern(term::Add, mkInt(1), mkInt(2))),
               intern(term::Eq, var(type::Int, 0), var(type::Int, 1)),
               unified));

  unified.n = 0;
  assert(!unify(intern(term::Eq, intern(term::Add, mkInt(1), mkInt(2)),
                       intern(term::Add, mkInt(1), mkInt(2))),
                intern(term::Eq, var(type::Real, 0), var(type::Real, 1)),
                unified));

  unified.n = 0;
  assert(unify(mk(term::Eq, mk(term::Call, addInt, mkInt(1), mkInt(2)),
                  mk(term::Call, addInt, mkInt(1), mkInt(2))),
               mk(term::Eq, var(type::Int, 0), var(type::Int, 1)), unified));

  unified.n = 0;
  assert(!unify(mk(term::Eq, mk(term::Call, addInt, mkInt(1), mkInt(2)),
                   mk(term::Call, addInt, mkInt(1), mkInt(2))),
                mk(term::Eq, var(type::Real, 0), var(type::Real, 1)), unified));

  unified.n = 0;
  assert(unify(intern(term::Eq, intern(term::Call, addInt, mkInt(1), mkInt(2)),
                      intern(term::Call, addInt, mkInt(1), mkInt(2))),
               intern(term::Eq, var(type::Int, 0), var(type::Int, 1)),
               unified));

  unified.n = 0;
  assert(!unify(intern(term::Eq, intern(term::Call, addInt, mkInt(1), mkInt(2)),
                       intern(term::Call, addInt, mkInt(1), mkInt(2))),
                intern(term::Eq, var(type::Real, 0), var(type::Real, 1)),
                unified));

  // make sure match correctly rejects terms with different types
  unified.n = 0;
  assert(match(var(type::Int, 0), mk(term::Add, mkInt(1), mkInt(2)), unified));

  unified.n = 0;
  assert(!match(var(type::Individual, 0), mk(term::Add, mkInt(1), mkInt(2)),
                unified));

  unified.n = 0;
  assert(match(mk(term::Eq, var(type::Int, 0), var(type::Int, 1)),
               mk(term::Eq, mk(term::Add, mkInt(1), mkInt(2)),
                  mk(term::Add, mkInt(1), mkInt(2))),
               unified));

  unified.n = 0;
  assert(!match(mk(term::Eq, var(type::Real, 0), var(type::Real, 1)),
                mk(term::Eq, mk(term::Add, mkInt(1), mkInt(2)),
                   mk(term::Add, mkInt(1), mkInt(2))),
                unified));

  unified.n = 0;
  assert(match(intern(term::Eq, var(type::Int, 0), var(type::Int, 1)),
               intern(term::Eq, intern(term::Add, mkInt(1), mkInt(2)),
                      intern(term::Add, mkInt(1), mkInt(2))),
               unified));

  unified.n = 0;
  assert(!match(intern(term::Eq, var(type::Real, 0), var(type::Real, 1)),
                intern(term::Eq, intern(term::Add, mkInt(1), mkInt(2)),
                       intern(term::Add, mkInt(1), mkInt(2))),
                unified));

  unified.n = 0;
  assert(match(mk(term::Eq, var(type::Int, 0), var(type::Int, 1)),
               mk(term::Eq, mk(term::Call, addInt, mkInt(1), mkInt(2)),
                  mk(term::Call, addInt, mkInt(1), mkInt(2))),
               unified));

  unified.n = 0;
  assert(!match(mk(term::Eq, var(type::Real, 0), var(type::Real, 1)),
                mk(term::Eq, mk(term::Call, addInt, mkInt(1), mkInt(2)),
                   mk(term::Call, addInt, mkInt(1), mkInt(2))),
                unified));

  unified.n = 0;
  assert(match(intern(term::Eq, var(type::Int, 0), var(type::Int, 1)),
               intern(term::Eq, intern(term::Call, addInt, mkInt(1), mkInt(2)),
                      intern(term::Call, addInt, mkInt(1), mkInt(2))),
               unified));

  unified.n = 0;
  assert(!match(intern(term::Eq, var(type::Real, 0), var(type::Real, 1)),
                intern(term::Eq, intern(term::Call, addInt, mkInt(1), mkInt(2)),
                       intern(term::Call, addInt, mkInt(1), mkInt(2))),
                unified));
}
#endif
