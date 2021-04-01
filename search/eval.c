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
    }
  }
  }
  return 0;
}

si eval(si a) { return a; }
