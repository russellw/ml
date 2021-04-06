#include "main.h"

static si stack[1 << 20];
static int stacki;

noret err(char *msg) {
  for (int i = 0; i < stacki; i++)
    print(stderr, stack[i]);
  fprintf(stderr, "%s\n", msg);
  stacktrace();
  exit(1);
}

static si match(si env, si a, si b) {
  if (tag(a) == t_cons && tag(b) == t_cons)
    for (; a != nil; a = tl(a)) {
      si a1 = hd(a);
      switch (keyword(hd(a1))) {
      case w_unquote:
        if (b == nil)
          return 0;
        env = cons(list3(mkeyword(w_let), hd(tl(a1)), hd(b)), env);
        b = tl(b);
        continue;
      case w_unquotes:
        env = cons(list3(mkeyword(w_let), hd(tl(a1)), b), env);
        b = nil;
        continue;
      }
      env = match(env, a1, hd(b));
      if (!env)
        return 0;
      b = tl(b);
    }
  return a == b ? env : 0;
}

static si quote(si env, si a) {
  if (tag(a) != t_cons)
    return a;
  vec v;
  vinit(&v);
  while (a != nil) {
    si b = hd(a);
    a = tl(a);
    switch (keyword(hd(b))) {
    case w_unquote:
      b = eval(env, hd(tl(b)));
      vpush(&v, b);
      continue;
    case w_unquotes:
      b = eval(env, hd(tl(b)));
      while (b != nil) {
        vpush(&v, hd(b));
        b = tl(b);
      }
      continue;
    }
    vpush(&v, quote(env, b));
  }
  return list(&v);
}

si eval(si env, si a) {
  if (stacki == sizeof stack / sizeof *stack)
    err("eval: stack overflow");
  stack[stacki++] = a;
  switch (tag(a)) {
  case t_cons: {
    if (a == nil)
      break;
    si op = hd(a);
    a = tl(a);
    switch (keyword(op)) {
    case s_add:
      a = add(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_can_get: {
      int found;
      get(eval(env, hd(a)), eval(env, hd(tl(a))), &found);
      a = mkint(found);
      break;
    }
    case s_div:
      a = div2(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_div_c:
      a = div_c(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_div_e:
      a = div_e(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_div_f:
      a = div_f(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_div_t:
      a = div_t2(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_eq:
      a = mkint(eval(env, hd(a)) == eval(env, hd(tl(a))));
      break;
    case s_isexact:
      switch (tag(eval(env, hd(a)))) {
      case t_int:
      case t_rat:
        a = mkint(1);
        break;
      default:
        a = mkint(0);
        break;
      }
      break;
    case s_isinexact:
      a = mkint(tag(eval(env, hd(a))) == t_float);
      break;
    case s_islist:
      a = mkint(tag(eval(env, hd(a))) == t_cons);
      break;
    case s_issym:
      a = mkint(tag(eval(env, hd(a))) == t_sym);
      break;
    case s_lambda:
      a = list3(env, hd(a), hd(tl(a)));
      break;
    case s_le:
      a = mkint(le(eval(env, hd(a)), eval(env, hd(tl(a)))));
      break;
    case s_lt:
      a = mkint(lt(eval(env, hd(a)), eval(env, hd(tl(a)))));
      break;
    case s_mul:
      a = mul(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_ne:
      a = mkint(eval(env, hd(a)) != eval(env, hd(tl(a))));
      break;
    case s_rem_c:
      a = rem_c(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_rem_e:
      a = rem_e(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_rem_f:
      a = rem_f(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_rem_t:
      a = rem_t(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case s_sub:
      a = sub(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case w_abs:
      a = abs1(eval(env, hd(a)));
      break;
    case w_and:
      for (; a != nil; a = tl(a))
        if (!istrue(eval(env, hd(a)))) {
          a = mkint(0);
          goto end;
        }
      a = mkint(1);
      break;
    case w_ceil:
      a = ceil1(eval(env, hd(a)));
      break;
    case w_cons:
      a = cons(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case w_floor:
      a = floor1(eval(env, hd(a)));
      break;
    case w_get: {
      int found;
      a = get(eval(env, hd(a)), eval(env, hd(tl(a))), &found);
      break;
    }
    case w_hd:
      a = hd(eval(env, hd(a)));
      break;
    case w_if:
      while (a != nil) {
        si x = eval(env, hd(a));
        a = tl(a);
        if (a == nil) {
          a = x;
          goto end;
        }
        if (istrue(x)) {
          a = eval(env, hd(a));
          goto end;
        }
        a = tl(a);
      }
      break;
    case w_let: {
      si key = hd(a);
      a = tl(a);
      si val = eval(env, hd(a));
      a = tl(a);
      si body = hd(a);
      a = eval(cons(list3(mkeyword(w_let), key, val), env), body);
      break;
    }
    case w_match: {
      si val = eval(env, hd(a));
      a = tl(a);
      while (a != nil) {
        si pat = hd(a);
        a = tl(a);
        si r = hd(a);
        a = tl(a);
        si env1 = match(env, pat, val);
        if (env1) {
          a = eval(env1, r);
          goto end;
        }
      }
      a = nil;
      break;
    }
    case w_minus:
      a = minus(eval(env, hd(a)));
      break;
    case w_not:
      a = mkint(!istrue(eval(env, hd(a))));
      break;
    case w_or:
      for (; a != nil; a = tl(a))
        if (istrue(eval(env, hd(a)))) {
          a = mkint(1);
          goto end;
        }
      a = mkint(0);
      break;
    case w_quote:
      a = quote(env, hd(a));
      break;
    case w_round:
      a = round1(eval(env, hd(a)));
      break;
    case w_tl:
      a = tl(eval(env, hd(a)));
      break;
    case w_trunc:
      a = trunc1(eval(env, hd(a)));
      break;
    default: {
      si f = eval(env, op);
      if (tag(f) != t_cons)
        err("eval: not a function");
      si fenv = hd(f);
      f = tl(f);
      si params = hd(f);
      f = tl(f);
      si body = hd(f);
      for (; params != nil; params = tl(params)) {
        fenv = cons(list3(mkeyword(w_let), hd(params), eval(env, hd(a))), fenv);
        a = tl(a);
      }
      a = eval(fenv, body);
      break;
    }
    }
    break;
  }
  case t_sym: {
    int found;
    a = get(env, a, &found);
    if (!found)
      err("eval: symbol not found");
    break;
  }
  }
end:
  stacki--;
  return a;
}
