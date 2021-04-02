enum {
  // SORT
  t_cons,
  t_float,
  t_int,
  t_rat,
  t_sym,
  ///

  t_max
};

// make a term
static si term(void *p, int t) {
  assert(p || t == t_cons);
  assert(0 <= t);
  assert(t < t_max);
  return (si)p + t;
}

#define nil term(0, t_cons)

// unpack a term
static int tag(si a) {
  si t = a & 7;
  assert(t < t_max);
  return t;
}

// SORT
static Cons *consp(si a) {
  assert(tag(a) == t_cons);
  return (Cons *)(a - t_cons);
}

static Float *floatp(si a) {
  assert(tag(a) == t_float);
  return (Float *)(a - t_float);
}

static Int *intp(si a) {
  assert(tag(a) == t_int);
  return (Int *)(a - t_int);
}

static Rat *ratp(si a) {
  assert(tag(a) == t_rat);
  return (Rat *)(a - t_rat);
}

static sym *symp(si a) {
  assert(tag(a) == t_sym);
  return (sym *)(a - t_sym);
}
///

static si keyword(si a) {
  // turn a symbol into a keyword number by subtracting the base of the keyword
  // array and dividing by the declared size of a symbol structure (which is
  // efficient as long as that size is a power of 2)

  // it's okay if the symbol is not a keyword; that just means the resulting
  // number will not correspond to any keyword and will not match any case in a
  // switch statement
  size_t i = (char *)symp(a) - (char *)keywords;
  return i / sizeof(sym);
}

// some obvious names e.g. 'div' and 'div_t' are taken
// SORT
si abs1(si a);
si add(si a, si b);
si ceil1(si a);
si div2(si a, si b);
si div_c(si a, si b);
si div_e(si a, si b);
si div_f(si a, si b);
si div_t2(si a, si b);
si exact(si a);
si floor1(si a);
si inexact(si a);
int istrue(si a);
int le(si a, si b);
int lt(si a, si b);
si minus(si a);
si mul(si a, si b);
void print(si a);
void println(si a);
si rem_c(si a, si b);
si rem_e(si a, si b);
si rem_f(si a, si b);
si rem_t(si a, si b);
si round1(si a);
si sub(si a, si b);
si trunc1(si a);
///
