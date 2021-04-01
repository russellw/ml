enum {
  // SORT
  t_float,
  t_int,
  t_rat,
  t_sym,
  ///

  t_max
};

// make a term
static si term(void *p, si t) {
  assert(p);
  assert(0 <= t);
  assert(t < t_max);
  return (si)p + t;
}

// unpack a term
static si tag(si a) {
  si t = a & 7;
  assert(t < t_max);
  return t;
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
