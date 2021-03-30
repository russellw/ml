#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

// temporary compound terms
compound *mk(si n) {
  auto r = (compound *)pool1.alloc(offsetof(compound, v) + n * sizeof(term));
  r->n = n;
  return r;
}

term mk(term op, const vec<term> &v) {
  auto n = v.n;
  auto r = mk(n);
  memcpy(r->v, v.p, n * sizeof *v.p);
  return tag(op, r);
}

term mk(term op, term a) {
  auto r = mk(1);
  r->v[0] = a;
  return tag(op, r);
}

term mk(term op, term a, term b) {
  auto r = mk(2);
  r->v[0] = a;
  r->v[1] = b;
  return tag(op, r);
}

term mk(term op, term a, term b, term c) {
  auto r = mk(3);
  r->v[0] = a;
  r->v[1] = b;
  r->v[2] = c;
  return tag(op, r);
}

// permanent/interned compound terms
namespace {
si cap = 0x1000;
si count;
compound **entries = (compound **)xcalloc(cap, sizeof *entries);

bool eq(const compound *x, const term *p, si n) {
  if (x->n != n)
    return 0;
  return !memcmp(x->v, p, n * sizeof *p);
}

si slot(compound **entries, si cap, const term *p, si n) {
  auto mask = cap - 1;
  auto i = XXH64(p, n * sizeof *p, 0) & mask;
  while (entries[i] && !eq(entries[i], p, n))
    i = (i + 1) & mask;
  return i;
}

void expand() {
  assert(isPow2(cap));
  auto cap1 = cap * 2;
  auto entries1 = (compound **)xcalloc(cap1, sizeof *entries);
  for (auto i = entries, e = entries + cap; i != e; ++i) {
    auto x = *i;
    if (x)
      entries1[slot(entries1, cap1, x->v, x->n)] = x;
  }
  free(entries);
  cap = cap1;
  entries = entries1;
}

compound *store(const term *p, si n) {
  auto r = (compound *)xmalloc(offsetof(compound, v) + n * sizeof *p);
  r->n = n;
  memcpy(r->v, p, n * sizeof *p);
  return r;
}

compound *put(const term *p, si n) {
  auto i = slot(entries, cap, p, n);
  if (entries[i])
    return entries[i];
  if (++count > cap * 3 / 4) {
    expand();
    i = slot(entries, cap, p, n);
  }
  return entries[i] = store(p, n);
}
} // namespace

void initTerms() {
  for (auto i = entries, e = entries + cap; i != e; ++i)
    free(*i);
  memset(entries, 0, cap * sizeof *entries);
}

term intern(term op, const vec<term> &v) { return tag(op, put(v.p, v.n)); }

term intern(term op, term a) { return tag(op, put(&a, 1)); }

term intern(term op, term a, term b) {
  term v[2];
  v[0] = a;
  v[1] = b;
  return tag(op, put(v, sizeof v / sizeof *v));
}

term intern(term op, term a, term b, term c) {
  term v[3];
  v[0] = a;
  v[1] = b;
  v[2] = c;
  return tag(op, put(v, sizeof v / sizeof *v));
}

// variables
namespace {
vec<term> boundVars;
} // namespace

void getFreeVars(term a, vec<term> &freeVars) {
  switch (tag(a)) {
  case term::All:
  case term::Exists: {
    auto old = boundVars.n;
    for (auto i = begin(a) + 1, e = end(a); i != e; ++i)
      boundVars.push_back(*i);
    getFreeVars(at(a, 0), freeVars);
    boundVars.n = old;
    return;
  }
  case term::Var:
    if (find(boundVars.p, boundVars.end(), a) != boundVars.end())
      return;
    if (find(freeVars.p, freeVars.end(), a) != freeVars.end())
      return;
    freeVars.push_back(a);
    return;
  }
  if (!isCompound(a))
    return;
  for (auto b : a)
    getFreeVars(b, freeVars);
}

#ifdef DEBUG
void ck(term a) {
  ck(typeof(a));
  if (isCompound(a)) {
    auto n = size(a);
    assert(0 < n);
    assert(n < 1000000);
    for (auto b : a)
      ck(b);
  }
  switch (tag(a)) {
  case term::Add:
  case term::DivE:
  case term::DivF:
  case term::DivT:
  case term::Le:
  case term::Lt:
  case term::Mul:
  case term::RemE:
  case term::RemF:
  case term::RemT:
  case term::Sub:
    assert(size(a) == 2);
    for (auto b : a)
      typeofNum(b);
    break;
  case term::Call:
    assert(1 < size(a));
    assert(tag(at(a, 0)) == term::Sym);
    break;
  case term::Ceil:
  case term::Floor:
  case term::IsInt:
  case term::IsRat:
  case term::Minus:
  case term::Round:
  case term::ToInt:
  case term::ToRat:
  case term::ToReal:
  case term::Trunc:
    assert(size(a) == 1);
    typeofNum(at(a, 0));
    break;
  case term::DistinctObj:
  case term::Sym:
    ck((sym *)rest(a));
    break;
  case term::Div:
    assert(size(a) == 2);
    for (auto b : a) {
      typeofNum(b);
      assert(typeof(b) != type::Int);
    }
    break;
  case term::Eq:
    assert(size(a) == 2);
    break;
  case term::Eqv:
  case term::Imp:
    assert(size(a) == 2);
    for (auto b : a)
      assert(typeof(b) == type::Bool);
    break;
  case term::Int:
    ck((Int *)rest(a));
    break;
  case term::Not:
    assert(size(a) == 1);
    assert(typeof(at(a, 0)) == type::Bool);
    break;
  case term::Rat:
  case term::Real:
    ck((Rat *)rest(a));
    break;
  case term::Var:
    assert(!isCompound(varType(a)));
    assert(0 <= vari(a));
    assert(vari(a) < 1000000);
    break;
  }
}
#endif

namespace {
bool needParens(term a, term parent) {
  switch (tag(a)) {
  case term::And:
  case term::Eqv:
  case term::Imp:
  case term::Or:
    switch (tag(parent)) {
    case term::All:
    case term::And:
    case term::Eqv:
    case term::Exists:
    case term::Imp:
    case term::Not:
    case term::Or:
      return 1;
    }
    break;
  }
  return 0;
}

// SORT
void infix(const char *op, term a, term parent) {
  auto parens = needParens(a, parent);
  if (parens)
    putchar('(');
  for (si i = 0, n = size(a); i != n; ++i) {
    if (i)
      printf("%s", op);
    print(at(a, i), a);
  }
  if (parens)
    putchar(')');
}

void quant(char op, term a) {
  printf("%c[", op);
  for (si i = 1, n = size(a); i != n; ++i) {
    if (i > 1)
      putchar(',');
    auto x = at(a, i);
    print(x);
    auto t = varType(x);
    if (t != type::Individual) {
      putchar(':');
      print(t);
    }
  }
  printf("]:");
  print(at(a, 0), a);
}

bool special(const char *s) {
  if (!isLower(*s))
    return 1;
  do
    if (!isWord[*s])
      return 1;
  while (*++s);
  return 0;
}
///
} // namespace

void print(term a, term parent) {
  switch (tag(a)) {
  case term::Add:
    printf("$sum");
    break;
  case term::All:
    quant('!', a);
    return;
  case term::And:
    infix(" & ", a, parent);
    return;
  case term::Call:
    print(at(a, 0), a);
    putchar('(');
    assert(size(a) > 1);
    for (si i = 1, n = size(a); i != n; ++i) {
      if (i > 1)
        putchar(',');
      print(at(a, i), a);
    }
    putchar(')');
    return;
  case term::Ceil:
    printf("$ceiling");
    break;
  case term::DistinctObj:
    quote('"', ((sym *)rest(a))->v);
    return;
  case term::Div:
    printf("$quotient");
    break;
  case term::DivE:
    printf("$quotient_e");
    break;
  case term::DivF:
    printf("$quotient_f");
    break;
  case term::DivT:
    printf("$quotient_t");
    break;
  case term::Eq:
    infix("=", a, parent);
    return;
  case term::Eqv:
    infix(" <=> ", a, parent);
    return;
  case term::Exists:
    quant('?', a);
    return;
  case term::False:
    printf("$false");
    return;
  case term::Floor:
    printf("$floor");
    break;
  case term::Imp:
    infix(" => ", a, parent);
    return;
  case term::Int:
    print(*((Int *)rest(a)));
    return;
  case term::IsInt:
    printf("$is_int");
    break;
  case term::IsRat:
    printf("$is_rat");
    break;
  case term::Le:
    printf("$lesseq");
    break;
  case term::Lt:
    printf("$less");
    break;
  case term::Minus:
    printf("$uminus");
    break;
  case term::Mul:
    printf("$product");
    break;
  case term::Not:
    putchar('~');
    print(at(a, 0), a);
    return;
  case term::Or:
    infix(" | ", a, parent);
    return;
  case term::Rat:
    print(*((Rat *)rest(a)));
    return;
  case term::Real:
    printf("%f", mpq_get_d(((Rat *)rest(a))->val));
    return;
  case term::RemE:
    printf("$remainder_e");
    break;
  case term::RemF:
    printf("$remainder_f");
    break;
  case term::RemT:
    printf("$remainder_t");
    break;
  case term::Round:
    printf("$round");
    break;
  case term::Sub:
    printf("$difference");
    break;
  case term::Sym: {
    auto s = ((sym *)rest(a))->v;
    if (special(s)) {
      quote('\'', s);
      return;
    }
    printf("%s", s);
    return;
  }
  case term::ToInt:
    printf("$to_int");
    break;
  case term::ToRat:
    printf("$to_rat");
    break;
  case term::ToReal:
    printf("$to_real");
    break;
  case term::True:
    printf("$true");
    return;
  case term::Trunc:
    printf("$truncate");
    break;
  case term::Var: {
    auto i = vari(a);
    if (i < 26) {
      putchar('A' + i);
      return;
    }
    printf("Z%zu", i - 25);
    return;
  }
  default:
    unreachable;
  }
  putchar('(');
  for (si i = 0, n = size(a); i != n; ++i) {
    if (i)
      putchar(',');
    print(at(a, i), a);
  }
  putchar(')');
}
