enum class szs {
  none,
#define X(s) s,
#include "szs.h"
  max
};

extern const char *szsNames[];

// SORT
extern clause *conjecture;
extern time_t deadline;
extern vec<clause *> problem;
///

#ifdef DEBUG
extern szs expected;
#endif

clause *input(vec<term> &neg, vec<term> &pos, how derived);
void initSolver();
