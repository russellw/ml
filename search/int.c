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
void mpz_qround(mpz_t q, mpz_t n, mpz_t d) {
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

//interned integers
static  si cap = 0x10;
static  si count;
static  Int **entries ;

static  size_t hash(Int *x) { return mpz_get_ui(x->val); }

static si eq(Int*x,Int*y){
	return!mpz_cmp(x->val,y->val);
}

static  si slot(Int **entries, si cap, Int *x) {
    si mask = cap - 1;
    si i = hash(x) & mask;
    while (entries[i] && !eq(entries[i],x))
      i = (i + 1) & mask;
    return i;
  }

static  void expand(void) {
    assert(ispow2(cap));
    si cap1 = cap * 2;
    Int** entries1 = xcalloc(cap1, sizeof *entries);
    for(si i=0;i<cap;i++){
      Int* x = entries[i];
      if (x)
        entries1[slot(entries1, cap1, x)] = x;
    }
    free(entries);
    cap = cap1;
    entries = entries1;
  }

static  Int *store(const Int *x) {
     Int * r =xmalloc(sizeof *x);
    *r = *x;
    return r;
  }

void init_ints(void) {
	entries = xcalloc(cap, sizeof *entries);
}

Int *intern_int(Int *x) {
    si i = slot(entries, cap, x);
    if (entries[i]) {
      mpz_clear(x->val);
      return entries[i];
    }
    if (++count > cap * 3 / 4) {
      expand();
      i = slot(entries, cap, x);
    }
    return entries[i] = store(x);
  }


