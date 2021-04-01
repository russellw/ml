enum {
  // SORT
  t_float,
  t_int,
  t_rat,
  t_sym,
  ///
};

#define term(p, t) ((si)(p) + (t))
#define tag(a) ((a)&7)

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
