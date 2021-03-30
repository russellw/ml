#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

const char *szsNames[] = {
    0,
#define X(s) #s,
#include "szs.h"
};

// SORT
clause *conjecture;
time_t deadline;
vec<clause *> problem;
///

#ifdef DEBUG
szs expected;
#endif

clause *input(vec<term> &neg, vec<term> &pos, how derived) {
  auto c = intern(neg, pos, derived);
  if (c)
    problem.push_back(c);
  return c;
}

void initSolver() {
  // SORT
  conjecture = 0;
  pool1.init();
  problem.n = 0;
  initClauses();
  initNums();
  initSyms();
  initTerms();
  ///
#ifdef DEBUG
  expected = szs::none;
#endif
}
