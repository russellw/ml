#include "main.h"

si eval(si env, si a) {
  if (tag(a) != t_cons)
    return a;
  si op = hd(a);
  a = tl(a);
  if (tag(op) != t_sym)
    err("eval: op is not a symbol");
  switch (keyword(op)) {
  case w_if: {
    si test = eval(env, hd(a));
    a = tl(a);
    if (!istrue(test))
      a = tl(a);
    return eval(env, hd(a));
  }
  case w_quote:
    return hd(a);
  }
  err("eval: operator not found");
}
