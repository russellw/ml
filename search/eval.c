#include "main.h"

static si stack[1 << 20];
static int stacki;

noret err(char *msg) {
  for (int i = 0; i < stacki; i++)
    print(stderr, stack[i]);
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

static int match(si a, si b, vec *frame) {
  if (tag(a) == t_cons && tag(b) == t_cons)
    while (a != nil) {
      si a1 = hd(a);
      a = tl(a);
      switch (keyword(hd(a1))) {
      case w_unquote:
        if (b == nil)
          return 0;
        vpush(frame, list3(mkeyword(w_val), hd(tl(a1)), hd(b)));
        b = tl(b);
        continue;
      case w_unquotes:
        vpush(frame, list3(mkeyword(w_val), hd(tl(a1)), b));
        b = nil;
        continue;
      }
      if (!match(a1, hd(b), frame))
        return 0;
      b = tl(b);
    }
  return a == b;
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
      a = list2(a, mkint(found));
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
    case w_match: {
      si val = eval(env, hd(a));
      a = tl(a);
      vec frame;
      vinit(&frame);
      while (a != nil) {
        si pat = hd(a);
        a = tl(a);
        si r = hd(a);
        a = tl(a);
        frame.n = 0;
        if (match(pat, val, &frame)) {
          a = eval(cons(list(&frame), env), r);
          goto end;
        }
      }
      vfree(&frame);
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
      vec frame;
      vinit(&frame);
      while (params != nil) {
        vpush(&frame, list3(mkeyword(w_val), hd(params), eval(env, hd(a))));
        params = tl(params);
        a = tl(a);
      }
      a = eval(cons(list(&frame), fenv), body);
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
