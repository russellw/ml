#include "main.h"

si eval(si env, si a) {
  if (tag(a) != t_cons)
    return a;
  si op = hd(a);
  a = tl(a);
  if (tag(op) != t_sym)
    err("eval: op is not a symbol");
  switch (keyword(op)) {
  case w_and:
    for (; a != nil; a = tl(a))
      if (!istrue(eval(env, hd(a))))
        return mkint(0);
    return mkint(1);
  case w_if: {
    si test = eval(env, hd(a));
    a = tl(a);
    if (!istrue(test))
      a = tl(a);
    return eval(env, hd(a));
  }
  case w_not:
    return mkint(!istrue(eval(env, hd(a))));
  case w_or:
    for (; a != nil; a = tl(a))
      if (istrue(eval(env, hd(a))))
        return mkint(1);
    return mkint(0);
  case w_quote:
    return hd(a);
  }
  err("eval: operator not found");
}
