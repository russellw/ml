#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

namespace {
// passive clauses are stored in a priority queue with smaller clauses first
si weight(term a) {
  if (isCompound(a)) {
    si n = 0;
    for (auto b : a)
      n += weight(b);
    return n;
  }
  return 1;
}

si weight(clause *c) {
  si n = 0;
  for (auto i = c->v, e = c->v + c->n; i != e; ++i)
    n += weight(*i);
  return n;
}

struct cmp {
  bool operator()(clause *c, clause *d) { return weight(c) > weight(d); }
};

priority_queue<clause *, vec<clause *>, cmp> passive;

// unification and matching procedures assume the two sides have distinct
// variable names, so when a given clause is selected, it must go through
// variable renaming before these procedures are applied; it would have been
// possible to redesign the procedures to eliminate the need for this, but the
// variable renaming makes the rest of the code simpler, and because it only
// needs to be done once each time around the outer loop, it is probably also
// more efficient
const uint64_t altVar = (uint64_t)1 << (atomicTypeBits + tagBits);

term altVars(term a) {
  if (tag(a) == term::Var)
    return term((uint64_t)a | altVar);
  if (!isCompound(a))
    return a;
  auto n = size(a);
  auto r = mk(n);
  for (si i = 0; i != n; ++i)
    r->v[i] = altVars(at(a, i));
  return tag(tag(a), r);
}

clause *altVars(clause *c) {
  auto n = c->n;
  auto d = (clause *)pool1.alloc(offsetof(clause, v) + n * sizeof(term));
  memcpy(d, c, offsetof(clause, v));
  for (si i = 0; i != n; ++i)
    d->v[i] = altVars(c->v[i]);
  return d;
}

// once new literals have been constructed from two input clauses with different
// variable namespaces, the variable names must be normalized, both to improve
// the ability of the system to detect duplicate terms and clauses, and to
// maintain the invariant that the variable names in active clauses are distinct
// from those in the alternative variable namespace
term normVars(term a, vec<pair<term, term>> &vars) {
  if (tag(a) == term::Var) {
    for (auto &i : vars)
      if (i.first == a)
        return i.second;
    auto b = var(varType(a), vars.n);
    vars.push_back(make_pair(a, b));
    return b;
  }
  if (!isCompound(a))
    return a;
  auto n = size(a);
  auto r = mk(n);
  for (si i = 0; i != n; ++i)
    r->v[i] = normVars(at(a, i), vars);
  return tag(tag(a), r);
}

// make a clause and if successful (not a tautology or duplicate) add it to the
// passive queue
vec<term> neg, pos;

void qclause(how derived) {
  vec<pair<term, term>> vars;
  for (auto &a : neg)
    a = normVars(a, vars);
  for (auto &a : pos)
    a = normVars(a, vars);

  auto c = intern(neg, pos, derived);
  neg.n = pos.n = 0;
  if (c)
    passive.push(c);
}

// inputs
clause *c;
term *ci;
term c0, c1;

clause *d;
term *di;
term d0, d1;

vec<pair<term, term>> unified;

/*
equality resolution
    c | c0 != c1
->
    c/s
where
    s = unify(c0, c1)
*/

// substitute and make new clause
void resolveq() {
  assert(!neg.n);
  for (auto i = c->v, e = c->v + c->nn; i != e; ++i)
    if (i != ci)
      neg.push_back(replace(*i, unified));

  assert(!pos.n);
  for (auto i = c->v + c->nn, e = c->v + c->n; i != e; ++i)
    pos.push_back(replace(*i, unified));

  qclause(how::resolve);
}

// for each negative equation
void resolve() {
  for (auto i = c->v, e = c->v + c->nn; i != e; ++i) {
    eqn ce(*i);
    unified.n = 0;
    if (unify(ce.left, ce.right, unified)) {
      ci = i;
      resolveq();
    }
  }
}

/*
equality factoring
    c | c0 = c1 | d0 = d1
->
    (c | c0 = c1 | c1 != d1)/s
where
    s = unify(c0, d0)
*/

bool equatable(term a, term b) {
  if (typeof(a) != typeof(b))
    return 0;
  if (typeof(a) == type::Bool)
    return a == term::True || b == term::True;
  return 1;
}

term equate(term a, term b) {
  assert(equatable(a, b));
  if (a == term::True)
    return b;
  if (b == term::True)
    return a;
  return mk(term::Eq, a, b);
}

// substitute and make new clause
void factorq() {
  if (!equatable(c1, d1))
    return;
  unified.n = 0;
  if (!unify(c0, d0, unified))
    return;

  assert(!neg.n);
  for (auto i = c->v, e = c->v + c->nn; i != e; ++i)
    neg.push_back(replace(*i, unified));
  neg.push_back(equate(replace(c1, unified), replace(d1, unified)));

  assert(!pos.n);
  for (auto i = c->v + c->nn, e = c->v + c->n; i != e; ++i)
    if (i != di)
      pos.push_back(replace(*i, unified));

  qclause(how::factor);
}

// for each positive equation (both directions) again
void factor1() {
  for (auto i = c->v + c->nn, e = c->v + c->n; i != e; ++i) {
    if (i == ci)
      continue;
    eqn de(*i);
    di = i;

    d0 = de.left;
    d1 = de.right;
    factorq();

    d0 = de.right;
    d1 = de.left;
    factorq();
  }
}

// for each positive equation (both directions)
void factor() {
  for (auto i = c->v + c->nn, e = c->v + c->n; i != e; ++i) {
    eqn ce(*i);
    ci = i;

    c0 = ce.left;
    c1 = ce.right;
    factor1();

    c0 = ce.right;
    c1 = ce.left;
    factor1();
  }
}

/*
superposition
    c | c0 = c1, d | d0(x) ?= d1
->
    (c | d | d0(c1) ?= d1)/s
where
    s = unify(c0, x)
    x not a variable
*/

// substitute and make new clause
void superpositionq(term d0c1) {
  assert(!neg.n);
  for (auto i = c->v, e = c->v + c->nn; i != e; ++i)
    neg.push_back(replace(*i, unified));
  for (auto i = d->v, e = d->v + d->nn; i != e; ++i)
    if (i != di)
      neg.push_back(replace(*i, unified));

  assert(!pos.n);
  for (auto i = c->v + c->nn, e = c->v + c->n; i != e; ++i)
    if (i != ci)
      pos.push_back(replace(*i, unified));
  for (auto i = d->v + d->nn, e = d->v + d->n; i != e; ++i)
    if (i != di)
      pos.push_back(replace(*i, unified));

  // negative and positive superposition
  auto &v = di < (d->v + d->nn) ? neg : pos;
  v.push_back(equate(replace(d0c1, unified), replace(d1, unified)));

  qclause(how::sp);
}

vec<si> position;

term splice(term x, si *i) {
  if (i == position.end())
    return c1;
  assert(isCompound(x));
  vec<term> v;
  v.insert(v.p, begin(x), end(x));
  auto j = *i++;
  v[j] = splice(v[j], i);
  return intern(tag(x), v);
}

void descend(term x) {
  if (tag(x) == term::Var)
    return;
  unified.n = 0;
  if (unify(c0, x, unified))
    superpositionq(splice(d0, position.p));
  if (isCompound(x))
    for (si j = 0, n = size(x); j != n; ++j) {
      position.push_back(j);
      descend(at(x, j));
      --position.n;
    }
}

// for each equation in d (both directions)
void superposition1() {
  if (c0 == term::True)
    return;
  for (auto i = d->v, e = d->v + d->n; i != e; ++i) {
    eqn de(*i);
    di = i;

    d0 = de.left;
    d1 = de.right;
    position.n = 0;
    descend(d0);

    d0 = de.right;
    d1 = de.left;
    position.n = 0;
    descend(d0);
  }
}

// for each positive equation in c (both directions)
void superposition() {
  for (auto i = c->v + c->nn, e = c->v + c->n; i != e; ++i) {
    eqn ce(*i);
    ci = i;

    c0 = ce.left;
    c1 = ce.right;
    superposition1();

    c0 = ce.right;
    c1 = ce.left;
    superposition1();
  }
}
} // namespace

szs saturate() {
  // passive clauses
  while (!passive.empty())
    passive.pop();
  for (auto c : problem)
    passive.push(c);

  // active clauses
  vec<clause *> active;

// saturation proof procedure tries to perform all possible derivations until
// it derives false
loop:
  while (!passive.empty()) {
    // given clause
    auto g = passive.top();
    passive.pop();

    // empty clause = derived false = unsatisfiable
    if (!g->n)
      return szs::Unsatisfiable;

    // are we out of time?
    if (deadline && time(0) >= deadline)
      return szs::Timeout;

    // alternate variables
    pool1.init();
    auto g1 = altVars(g);

    // this is the Discount loop (in which only active clauses participate in
    // subsumption checks); in tests, it performed slightly better than the
    // alternative Otter loop (in which passive clauses also participate)

    // forward subsumption
    for (auto h : active) {
      if (h->subsumed)
        continue;
      if (subsumes(h, g1))
        goto loop;
    }

    // backward subsumption
    for (auto h : active) {
      if (h->subsumed)
        continue;
      if (subsumes(g1, h))
        h->subsumed = 1;
    }

    // add g to active clauses before inference because we will sometimes need
    // to combine g with (the alternate variable version of) itself
    active.push_back(g);

    // infer
    c = g1;
    resolve();
    factor();
    for (auto h : active) {
      if (h->subsumed)
        continue;

      c = g1;
      d = h;
      superposition();

      c = h;
      d = g1;
      superposition();
    }
  }

  // if a complete saturation proof procedure finds no more possible
  // derivations, then the problem is satisfiable; in practice, this almost
  // never happens for nontrivial problems, but serves as a good way to test the
  // completeness of the prover on some trivial problems. however, if
  // completeness was lost for any reason, such as having to discard some
  // clauses because they were too big, then report failure
  return complete ? szs::Satisfiable : szs::GaveUp;
}
