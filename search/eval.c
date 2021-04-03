#include "main.h"

static si apply(si f, si args) {
  si env = hd(f);
  f = tl(f);
  si params = hd(f);
  f = tl(f);
  si body = hd(f);
  env = cons(zip(params, args), env);
  return eval(env, body);
}

si eval(si env, si a) {
  // symbols are names to be looked up
  if (tag(a) == t_sym) {
    si pair = get(env, a);
    if (pair == nil)
      err("eval: symbol not found");
    return hd(tl(pair));
  }

  // other atoms evaluate to themselves
  if (tag(a) != t_cons)
    return a;

  // lists are expressions to be evaluated based on the first element
  si op = hd(a);
  a = tl(a);
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
  si f = eval(env, op);
  return apply(f, a);
}
