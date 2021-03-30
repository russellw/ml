enum class how : char {
  none,
#define X(s) s,
#include "how.h"
};

extern const char *howNames[];

struct clause {
  // is this a first-order formula rather than an actual clause?
  // a first-order formula is represented like a clause with just one positive
  // literal, but the literal can be any first-order predicate
  bool fof;

  // set if this clause has been subsumed and should be henceforth ignored (or
  // perhaps eventually garbage collected)
  bool subsumed;

  // how was it derived?
  how derived;

  // number of negative and total literals
  // the literals are laid out in an array, negative then positive
  uint16_t nn, n;
  si np() { return n - nn; }

  // the majority of ways for a clause to be made, result in it deriving from
  // zero or one other clauses, but the majority of clauses will be made by the
  // superposition rule, so derived from two other clauses
  clause *from[2];

  // literals
  term v[0];
};

extern bool complete;

clause *mk(term a, how derived, clause *from = 0);
clause *intern(vec<term> &neg, vec<term> &pos, how derived, clause *from = 0,
               clause *from1 = 0);

const char *getFile(const clause *c);
const char *getName(const clause *c);
void setFile(clause *c, const char *file);
void setName(clause *c, const char *name);

void getProof(clause *c, vec<clause *> &proof);

void initClauses();

#ifdef DEBUG
void ck(clause *c);
#else
inline void ck(clause *c) {}
#endif

void print(clause *c);
