#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

const char *howNames[] = {
    0,
#define X(s) #s,
#include "how.h"
};

bool complete;

// uninterned formulas
namespace {
pool<> formulas;
}

clause *mk(term a, how derived, clause *from) {
  auto r = (clause *)formulas.alloc(offsetof(clause, v) + sizeof a);
  memset(r, 0, offsetof(clause, v));
  r->fof = 1;
  r->derived = derived;
  r->n = 1;
  r->from[0] = from;
  r->from[1] = 0;
  r->v[0] = a;
  return r;
}

// interned clauses
namespace {
si cap = 0x1000;
si count;
clause **entries = (clause **)xcalloc(cap, sizeof *entries);

bool eq(const clause *c, const term *p, si nn, si n) {
  if (c->nn != nn)
    return 0;
  if (c->n != n)
    return 0;
  return !memcmp(c->v, p, n * sizeof *p);
}

si slot(clause **entries, si cap, const term *p, si nn, si n) {
  auto mask = cap - 1;
  auto i = XXH64(p, n * sizeof *p, nn) & mask;
  while (entries[i] && !eq(entries[i], p, nn, n))
    i = (i + 1) & mask;
  return i;
}

void expand() {
  assert(isPow2(cap));
  auto cap1 = cap * 2;
  auto entries1 = (clause **)xcalloc(cap1, sizeof *entries);
  for (auto i = entries, e = entries + cap; i != e; ++i) {
    auto c = *i;
    if (c)
      entries1[slot(entries1, cap1, c->v, c->nn, c->n)] = c;
  }
  free(entries);
  cap = cap1;
  entries = entries1;
}
} // namespace

clause *intern(vec<term> &neg, vec<term> &pos, how derived, clause *from,
               clause *from1) {
  // simplify literals e.g. replace x=x with true; this must be done before the
  // subsequent steps
  for (auto &a : neg)
    a = simplify(a);
  for (auto &a : pos)
    a = simplify(a);

  // remove redundancy; do this before sorting, to make sorting faster
  neg.erase(remove(neg.p, neg.end(), term::True), neg.end());
  pos.erase(remove(pos.p, pos.end(), term::False), pos.end());

  // sort the literals. it's not that the order is meaningful, but that sorting
  // them into canonical order (even if that order is different in each run due
  // to address space layout randomization) makes it possible to detect
  // duplicate clauses that vary only by permutation of literals
  sort(neg.p, neg.end());
  sort(pos.p, pos.end());

  // remove duplicate literals (must be done after sorting, because in order to
  // run in linear time, std::unique assumes duplicates will be consecutive)
  neg.erase(unique(neg.p, neg.end()), neg.end());
  pos.erase(unique(pos.p, pos.end()), pos.end());

  // gather literals
  auto nn = neg.n;
  auto np = pos.n;
  auto n = nn + np;
  if (n > 0xffff) {
    // if the number of literals would exceed 16-bit count, discard the clause.
    // in principle this breaks completeness, though in practice it is unlikely
    // such a clause would contribute anything useful to the proof search anyway
    complete = 0;
    return 0;
  }
  neg.resize(n);
  auto p = neg.p;
  memcpy(p + nn, pos.p, np * sizeof *p);

  // check for tautology (could be done earlier, but as the only step that takes
  // quadratic time, it is here being done after the efforts to reduce the
  // number of literals)
  for (auto i = p, e = p + nn; i != e; ++i)
    if (*i == term::False)
      return 0;
  for (auto i = p + nn, e = p + n; i != e; ++i)
    if (*i == term::True)
      return 0;
  for (auto i = p, e = p + nn; i != e; ++i) {
    auto a = *i;
    for (auto j = p + nn, e = p + n; j != e; ++j)
      if (a == *j)
        return 0;
  }

  // check for an identical existing clause; unlike some other datatypes, we
  // return null rather than just returning a pointer to the existing object,
  // because creating a new clause is a significant action; caller may need to
  // know whether to go ahead and add the new clause to various containers
  auto i = slot(entries, cap, p, nn, n);
  if (entries[i])
    return 0;
  if (++count > cap * 3 / 4) {
    expand();
    i = slot(entries, cap, p, nn, n);
  }

  // make new clause
  auto c = entries[i] = (clause *)xmalloc(offsetof(clause, v) + n * sizeof *p);
  memset(c, 0, offsetof(clause, v));
  c->derived = derived;
  c->nn = nn;
  c->n = n;
  c->from[0] = from;
  c->from[1] = from1;
  memcpy(c->v, p, n * sizeof *p);
  return c;
}

namespace {
unordered_map<const clause *, const char *> clauseFiles;
unordered_map<const clause *, const char *> clauseNames;
} // namespace

const char *getFile(const clause *c) { return clauseFiles[c]; }

const char *getName(const clause *c) {
  auto name = clauseNames[c];
  return name ? name : "?";
}

void setFile(clause *c, const char *file) {
  if (c)
    clauseFiles[c] = file;
}

void setName(clause *c, const char *name) {
  if (c)
    clauseNames[c] = name;
}

void getProof(clause *c, vec<clause *> &proof) {
  if (!c)
    return;
  if (find(proof.p, proof.end(), c) != proof.end())
    return;
  proof.push_back(c);
  getProof(c->from[0], proof);
  getProof(c->from[1], proof);
}

void initClauses() {
  // SORT
  clauseFiles.clear();
  clauseNames.clear();
  complete = 1;
  count = 0;
  formulas.init();
  ///

  for (auto i = entries, e = entries + cap; i != e; ++i) {
    free(*i);
    *i = 0;
  }
}

#ifdef DEBUG
void ck(clause *c) {
  ckPtr(c);
  assert(c->nn <= c->n);
  for (auto i = c->v, e = c->v + c->n; i != e; ++i)
    ck(*i);
  assert(!c->from[1] || c->from[0]);
  si nfrom = 0;
  for (si i = 0; i != sizeof c->from / sizeof c; ++i)
    if (c->from[i]) {
      ckPtr(c->from[0]);
      ++nfrom;
    }
  switch (c->derived) {
  case how::cnf:
  case how::factor:
  case how::negate:
  case how::resolve:
    assert(nfrom == 1);
    break;
  case how::none:
    assert(nfrom == 0);
    break;
  case how::sp:
    assert(nfrom == 2);
    break;
  default:
    unreachable;
  }
}
#endif

void print(clause *c) {
  if (!c) {
    printf("null");
    return;
  }
  printf(c->fof ? "fof" : "cnf");

  // name
  printf("(%s, ", getName(c));

  // role
  if (c == conjecture)
    printf("conjecture");
  else if (c->derived == how::negate)
    printf("negated_conjecture");
  else
    printf("plain");
  printf(", ");

  // literals
  if (c->n)
    for (si i = 0, n = c->n; i != n; ++i) {
      if (i)
        printf(" | ");
      if (i < c->nn)
        putchar('~');
      print(c->v[i]);
    }
  else
    printf("$false");
  printf(", ");

  // source
  auto file = getFile(c);
  if (file) {
    printf("file(");
    quote('\'', basename(file));
    printf(",%s", getName(c));
  } else if (*c->from) {
    printf("inference(%s,[status(", howNames[(si)c->derived]);
    if (*c->from == conjecture)
      printf("ceq");
    else
      printf("thm");
    printf(")],[%s", getName(*c->from));
    if (c->from[1])
      printf(",%s", getName(c->from[1]));
    putchar(']');
  } else
    printf("introduced(definition");
  printf(")).");
}
