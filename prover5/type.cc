#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

// atomic types
const char *typeNames[atomicTypes];

namespace {
struct init {
  init() {
    typeNames[(si)type::Bool] = "$o";
    typeNames[(si)type::Individual] = "$i";
    typeNames[(si)type::Int] = "$int";
    typeNames[(si)type::Rat] = "$rat";
    typeNames[(si)type::Real] = "$real";
  }
} init1;

si atoms = (si)type::max;
} // namespace

type internType(sym *name) {
  if (name->thisType != type::none)
    return name->thisType;
  if (atoms >= atomicTypes)
    err("too many atomic types");
  typeNames[atoms] = name->v;
  return name->thisType = (type)atoms++;
}

// compound types
namespace {
si cap = 0x10;
si count;
tcompound **entries = (tcompound **)xcalloc(cap, sizeof *entries);

bool eq(const tcompound *t, const type *p, si n) {
  if (t->n != n)
    return 0;
  return !memcmp(t->v, p, n * sizeof *p);
}

si slot(tcompound **entries, si cap, const type *p, si n) {
  auto mask = cap - 1;
  auto i = XXH64(p, n * sizeof *p, 0) & mask;
  while (entries[i] && !eq(entries[i], p, n))
    i = (i + 1) & mask;
  return i;
}

void expand() {
  assert(isPow2(cap));
  auto cap1 = cap * 2;
  auto entries1 = (tcompound **)xcalloc(cap1, sizeof *entries);
  for (auto i = entries, e = entries + cap; i != e; ++i) {
    auto t = *i;
    if (t)
      entries1[slot(entries1, cap1, t->v, t->n)] = t;
  }
  free(entries);
  cap = cap1;
  entries = entries1;
}

tcompound *store(const type *p, si n) {
  auto r = (tcompound *)mmalloc(offsetof(tcompound, v) + n * sizeof *p);
  r->n = n;
  memcpy(r->v, p, n * sizeof *p);
  return r;
}

type put(const type *p, si n) {
  auto i = slot(entries, cap, p, n);
  if (entries[i])
    return type(si(entries[i]));
  if (++count > cap * 3 / 4) {
    expand();
    i = slot(entries, cap, p, n);
  }
  return type(si(entries[i] = store(p, n)));
  //  return type(si(entries[i]));
}
} // namespace

type internType(const vec<type> &v) {
  if (v.n > 0xffff)
    throw "type too complex";
  return put(v.p, v.n);
}

type internType(type rt, type param1) {
  type v[2];
  v[0] = rt;
  v[1] = param1;
  return put(v, sizeof v / sizeof *v);
}

type internType(type rt, type param1, type param2) {
  vec<type> v(3);
  v[0] = rt;
  v[1] = param1;
  v[2] = param2;
  return internType(v);
}

// etc
void defaultType(type t, term a) {
  assert(!isCompound(t));
  switch (tag(a)) {
  case term::Call: {
    auto op = at(a, 0);
    assert(tag(op) == term::Sym);
    auto s = (sym *)rest(op);
    auto t = s->t;
    if (t != type::none)
      break;
    auto n = size(a);
    vec<type> v(n);
    v[0] = t;
    for (si i = 1; i != n; ++i) {
      auto u = typeof(at(a, i));
      assert(u != type::none);
      v[i] = u;
    }
    s->t = internType(v);
    break;
  }
  case term::Sym: {
    auto s = (sym *)rest(a);
    if (s->t == type::none)
      s->t = t;
    break;
  }
  }
}

void requireType(type t, term a) {
  defaultType(t, a);
  if (t != typeof(a))
    throw "type mismatch";
}

type typeof(term a) {
  switch (tag(a)) {
  case term::All:
  case term::And:
  case term::Eq:
  case term::Eqv:
  case term::Exists:
  case term::IsInt:
  case term::IsRat:
  case term::Le:
  case term::Lt:
  case term::Not:
  case term::Or:
    return type::Bool;
  case term::Call: {
    auto op = at(a, 0);
    assert(tag(op) == term::Sym);
    auto s = (sym *)rest(op);
    auto t = s->t;
    if (t == type::none)
      return t;
    assert(isCompound(t));
    auto p = tcompoundp(t);
    assert(size(a) == p->n);
    return p->v[0];
  }
  case term::DistinctObj:
    return type::Individual;
  case term::False:
  case term::True:
    return type::Bool;
  case term::Int:
    return type::Int;
  case term::Rat:
    return type::Rat;
  case term::Real:
    return type::Real;
  case term::Sym:
    return ((sym *)rest(a))->t;
  case term::ToInt:
    return type::Int;
  case term::ToRat:
    return type::Rat;
  case term::ToReal:
    return type::Real;
  case term::Var:
    return varType(a);
  }
  // many arithmetic operations return the type of their first operand
  assert(isCompound(a));
  return typeof(at(a, 0));
}

type typeofNum(term a) {
  auto t = typeof(a);
  switch (t) {
  case type::Int:
  case type::Rat:
  case type::Real:
    return t;
  }
  throw "expected number term";
}

#ifdef DEBUG
void ck(type t) {
  if (!isCompound(t)) {
    auto i = (si)t;
    if (!i)
      return;
    ckStr(typeNames[i]);
    return;
  }
  auto p = tcompoundp(t);
  ckPtr(p);
  auto n = p->n;
  assert(1 < n);
  for (si i = 0; i != n; ++i) {
    assert(p->v[i] != type::none);
    ck(p->v[i]);
  }
}
#endif

void print(type t) {
  auto i = (si)t;
  if (i < atomicTypes) {
    auto s = typeNames[i];
    assert(s);
    printf("%s", s);
    return;
  }
  auto p = tcompoundp(t);
  auto n = p->n;
  for (si i = 1; i != n; ++i) {
    if (i > 1)
      printf(" > ");
    print(p->v[i]);
  }
  printf(" > ");
  print(*p->v);
}
