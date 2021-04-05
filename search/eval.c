#include "main.h"

si apply(si f, si args) {
  si env = hd(f);
  f = tl(f);
  si params = hd(f);
  f = tl(f);
  si body = hd(f);
  env = cons(zip(params, args), env);
  return eval(env, body);
}

si eval(si env, si a0) {
  si a = a0;

  // symbols are names to be looked up
  if (tag(a) == t_sym) {
    int found;
    si val = get(env, a, &found);
    if (found)
      return val;
    print(stderr, a0);
    err("eval: symbol not found");
  }

  // other atoms evaluate to themselves
  if (tag(a) != t_cons)
    return a;

  // lists are expressions to be evaluated based on the first element
  si op = hd(a);
  a = tl(a);

  // known operator
  switch (keyword(op)) {
  case s_add:
    return add(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_div:
    return div2(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_div_c:
    return div_c(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_div_e:
    return div_e(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_div_f:
    return div_f(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_div_t:
    return div_t2(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_eq:
    return mkint(eval(env, hd(a)) == eval(env, hd(tl(a))));
  case s_lambda:
    return list3(env, hd(a), hd(tl(a)));
  case s_le:
    return mkint(le(eval(env, hd(a)), eval(env, hd(tl(a)))));
  case s_lt:
    return mkint(lt(eval(env, hd(a)), eval(env, hd(tl(a)))));
  case s_mul:
    return mul(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_ne:
    return mkint(eval(env, hd(a)) != eval(env, hd(tl(a))));
  case s_rem_c:
    return rem_c(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_rem_e:
    return rem_e(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_rem_f:
    return rem_f(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_rem_t:
    return rem_t(eval(env, hd(a)), eval(env, hd(tl(a))));
  case s_sub:
    return sub(eval(env, hd(a)), eval(env, hd(tl(a))));
  case w_abs:
    return abs1(eval(env, hd(a)));
  case w_and:
    for (; a != nil; a = tl(a))
      if (!istrue(eval(env, hd(a))))
        return mkint(0);
    return mkint(1);
  case w_ceil:
    return ceil1(eval(env, hd(a)));
  case w_floor:
    return floor1(eval(env, hd(a)));
  case w_if: {
    si test = eval(env, hd(a));
    a = tl(a);
    if (!istrue(test))
      a = tl(a);
    return eval(env, hd(a));
  }
  case w_minus:
    return minus(eval(env, hd(a)));
  case w_not:
    return mkint(!istrue(eval(env, hd(a))));
  case w_or:
    for (; a != nil; a = tl(a))
      if (istrue(eval(env, hd(a))))
        return mkint(1);
    return mkint(0);
  case w_quote:
    return hd(a);
  case w_round:
    return round1(eval(env, hd(a)));
  case w_trunc:
    return trunc1(eval(env, hd(a)));
  }

  // apply a function
  si f = eval(env, op);
  if (tag(f) != t_cons) {
    print(stderr, a0);
    err("eval: not a function");
  }
  si fenv = hd(f);
  f = tl(f);
  si params = hd(f);
  f = tl(f);
  si body = hd(f);
  vec frame;
  vinit(&frame);
  while (params != nil) {
    vpush(&frame, list2(hd(params), eval(env, hd(a))));
    params = tl(params);
    a = tl(a);
  }
  return eval(cons(list(&frame), fenv), body);
}
