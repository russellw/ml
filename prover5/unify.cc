#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

bool match(term a, term b, vec<pair<term, term>> &r) {
  // equal
  if (a == b)
    return 1;

  // type mismatch
  if (typeof(a) != typeof(b))
    return 0;

  // variable
  if (tag(a) == term::Var) {
    // existing mapping
    for (auto &i : r) {
      if (i.first == a)
        return i.second == b;
    }

    // new mapping
    r.push_back(make_pair(a, b));
    return 1;
  }

  // atoms
  if (!isCompound(a))
    return 0;
  if (tag(a) != tag(b))
    return 0;

  // compounds
  auto n = size(a);
  if (n != size(b))
    return 0;
  if (tag(a) == term::Call) {
    if (at(a, 0) != at(b, 0))
      return 0;
    for (si i = 1; i != n; ++i)
      if (!match(at(a, i), at(b, i), r))
        return 0;
    return 1;
  }
  for (si i = 0; i != n; ++i)
    if (!match(at(a, i), at(b, i), r))
      return 0;
  return 1;
}

namespace {
bool occurs(term a, term b, const vec<pair<term, term>> &r) {
  assert(tag(a) == term::Var);
  if (tag(b) == term::Var) {
    if (a == b)
      return 1;
    for (auto &i : r)
      if (i.first == b)
        return occurs(a, i.second, r);
  }
  if (!isCompound(b))
    return 0;
  for (auto x : b)
    if (occurs(a, x, r))
      return 1;
  return 0;
}

bool unifyVar(term a, term b, vec<pair<term, term>> &r) {
  assert(tag(a) == term::Var);
  assert(typeof(a) == typeof(b));

  // existing mappings
  for (auto &i : r) {
    if (i.first == a)
      return unify(i.second, b, r);
    if (i.first == b)
      return unify(a, i.second, r);
  }

  // occurs check
  if (occurs(a, b, r))
    return 0;

  // new mapping
  r.push_back(make_pair(a, b));
  return 1;
}
} // namespace

bool unify(term a, term b, vec<pair<term, term>> &r) {
  // equal
  if (a == b)
    return 1;

  // type mismatch
  if (typeof(a) != typeof(b))
    return 0;

  // variables
  if (tag(a) == term::Var)
    return unifyVar(a, b, r);
  if (tag(b) == term::Var)
    return unifyVar(b, a, r);

  // atoms
  if (!isCompound(a))
    return 0;
  if (tag(a) != tag(b))
    return 0;

  // compounds
  auto n = size(a);
  if (n != size(b))
    return 0;
  if (tag(a) == term::Call) {
    if (at(a, 0) != at(b, 0))
      return 0;
    for (si i = 1; i != n; ++i)
      if (!unify(at(a, i), at(b, i), r))
        return 0;
    return 1;
  }
  for (si i = 0; i != n; ++i)
    if (!unify(at(a, i), at(b, i), r))
      return 0;
  return 1;
}

term replace(term a, const vec<pair<term, term>> &unified) {
  if (tag(a) == term::Var)
    for (auto &i : unified)
      if (i.first == a)
        return replace(i.second, unified);
  if (!isCompound(a))
    return a;
  auto n = size(a);
  auto r = mk(n);
  for (si i = 0; i != n; ++i)
    r->v[i] = replace(at(a, i), unified);
  return tag(tag(a), r);
}
