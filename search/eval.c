#include "main.h"

si add(si a, si b) {
  switch (tag(a)) {
  case t_float: {
    Float *a1 = floatp(a);
    switch (tag(b)) {
    case t_float: {
      Float *b1 = floatp(b);
      return term(intern_float(a1->val + b1->val), t_float);
    }
    case t_int: {
      Int *b1 = intp(b);
      return term(intern_float(a1->val + mpz_get_d(b1->val)), t_float);
    }
    }
  }
  case t_int: {
    Int *a1 = intp(a);
    switch (tag(b)) {
    case t_int: {
      Int *b1 = intp(b);
      Int r;
      mpz_init(r.val);
      mpz_add(r.val, a1->val, b1->val);
      return term(intern_int(&r), t_int);
    }
    }
  }
  }
  return 0;
}

si eval(si a) { return a; }
