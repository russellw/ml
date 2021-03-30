#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

void mpz_ediv_q(mpz_t q, mpz_t n, mpz_t d) {
  mpz_t r;
  mpz_init(r);
  mpz_ediv_r(r, n, d);
  mpz_sub(q, n, r);
  mpz_clear(r);
  mpz_tdiv_q(q, q, d);
}

void mpz_ediv_r(mpz_t r, mpz_t n, mpz_t d) {
  mpz_tdiv_r(r, n, d);
  if (mpz_sgn(r) < 0) {
    mpz_t dabs;
    mpz_init(dabs);
    mpz_abs(dabs, d);
    mpz_add(r, r, dabs);
    mpz_clear(dabs);
  }
}

// calculate q = n/d, assuming common factors have already been canceled out,
// and applying bankers rounding
void round(mpz_t q, mpz_t n, mpz_t d) {
  // if we are dividing by 2, the result could be exactly halfway between two
  // integers, so need special case to apply bankers rounding
  if (!mpz_cmp_ui(d, 2)) {
    // floored division by 2 (this corresponds to arithmetic shift right one
    // bit)
    mpz_fdiv_q_2exp(q, n, 1);
    // if it was an even number before the division, the issue doesn't arise; we
    // already have the exact answer
    if (!mpz_tstbit(n, 0))
      return;
    // if it's an even number after the division, we are already on the nearest
    // even integer, so we don't need to do anything else
    if (!mpz_tstbit(q, 0))
      return;
    // need to adjust by one to land on an even integer, but which way? floored
    // division rounded down, so we need to go up
    mpz_add_ui(q, q, 1);
    return;
  }
  // we are not dividing by 2, so cannot end up exactly halfway between two
  // integers, and merely need to add half the denominator to the numerator
  // before dividing
  mpz_t d2;
  mpz_init(d2);
  mpz_fdiv_q_2exp(d2, d, 1);
  mpz_add(q, n, d2);
  mpz_clear(d2);
  mpz_fdiv_q(q, q, d);
}

namespace {
template <class T> struct bank {
  si cap = 0x10;
  si count;
  T **entries = (T **)xcalloc(cap, sizeof *entries);

  si slot(T **entries, si cap, const T &x) {
    auto mask = cap - 1;
    auto i = x.hash() & mask;
    while (entries[i] && !entries[i]->eq(x))
      i = (i + 1) & mask;
    return i;
  }

  void expand() {
    assert(isPow2(cap));
    auto cap1 = cap * 2;
    auto entries1 = (T **)xcalloc(cap1, sizeof *entries);
    for (auto i = entries, e = entries + cap; i != e; ++i) {
      auto x = *i;
      if (x)
        entries1[slot(entries1, cap1, *x)] = x;
    }
    free(entries);
    cap = cap1;
    entries = entries1;
  }

  T *store(const T &x) {
    auto r = (T *)xmalloc(sizeof x);
    *r = x;
    return r;
  }

  T *put(T &x) {
    auto i = slot(entries, cap, x);
    if (entries[i]) {
      x.clear();
      return entries[i];
    }
    if (++count > cap * 3 / 4) {
      expand();
      i = slot(entries, cap, x);
    }
    return entries[i] = store(x);
  }
};

bank<Int> ints;
bank<Rat> rats;
} // namespace

Int *intern(Int &x) { return ints.put(x); }

Rat *intern(Rat &x) {
  mpq_canonicalize(x.val);
  return rats.put(x);
}

void initNums() {
  for (auto i = ints.entries, e = ints.entries + ints.cap; i != e; ++i) {
    auto x = *i;
    if (x) {
      mpz_clear(x->val);
      free(x);
    }
  }
  memset(ints.entries, 0, ints.cap * sizeof *ints.entries);

  for (auto i = rats.entries, e = rats.entries + rats.cap; i != e; ++i) {
    auto x = *i;
    if (x) {
      mpq_clear(x->val);
      free(x);
    }
  }
  memset(rats.entries, 0, rats.cap * sizeof *rats.entries);
}

#ifdef DEBUG
void ck(const Int *x) { ckPtr(x); }

void ck(const Rat *x) {
  ckPtr(x);
  ckPtr(mpq_numref(x->val));
  ckPtr(mpq_denref(x->val));
  assert(mpz_cmp_ui(mpq_denref(x->val), 0) > 0);
}
#endif

void print(const Int &x) { mpz_out_str(stdout, 10, x.val); }

void print(const Rat &x) {
  mpq_out_str(stdout, 10, x.val);
  if (!mpz_cmp_ui(mpq_denref(x.val), 1))
    printf("/1");
}
