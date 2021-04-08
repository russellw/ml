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

static int matched(si key, si val, vec *records) {
  for (si i = 0; i < records->n; i++) {
    si record = records->p[i];
    assert(hd(record) == mkeyword(w_let));
    record = tl(record);
    if (hd(record) == key) {
      record = tl(record);
      return hd(record) == val;
    }
  }
  vpush(records, list3(mkeyword(w_let), key, val));
  return 1;
}

static int match(si a, si b, vec *records) {
  if (tag(a) == t_cons && tag(b) == t_cons)
    for (; a != nil; a = tl(a)) {
      si a1 = hd(a);
      switch (keyword(hd(a1))) {
      case w_unquote:
        if (b == nil)
          return 0;
        a1 = tl(a1);
        if (!matched(hd(a1), hd(b), records))
          return 0;
        b = tl(b);
        continue;
      case w_unquotes:
        a1 = tl(a1);
        if (!matched(hd(a1), b, records))
          return 0;
        b = nil;
        continue;
      }
      if (!match(a1, hd(b), records))
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
    case w_do:
      a = evals(env, a);
      break;
    case w_eval:
      a = eval(eval(env, hd(a)), eval(env, hd(tl(a))));
      break;
    case w_floor:
      a = floor1(eval(env, hd(a)));
      break;
    case w_get: {
      int found;
      a = get(eval(env, hd(a)), eval(env, hd(tl(a))), &found);
      break;
    }
    case w_has: {
      int found;
      get(eval(env, hd(a)), eval(env, hd(tl(a))), &found);
      a = mkint(found);
      break;
    }
    case w_hd:
      a = hd(eval(env, hd(a)));
      break;
    case w_here:
      a = env;
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
    case w_letrec: {
      vec record;
      vinit(&record);
      si b;
      for (;;) {
        b = hd(a);
        a = tl(a);
        if (a == nil)
          break;
        vpush(&record, b);
        vpush(&record, hd(a));
        a = tl(a);
        vpush(&record, hd(a));
        a = tl(a);
      }
      a = eval(cons(cons(mkeyword(w_letrec), list(&record)), env), b);
      break;
    }
    case w_list: {
      vec v;
      vinit(&v);
      for (; a != nil; a = tl(a))
        vpush(&v, eval(env, hd(a)));
      a = list(&v);
      break;
    }
    case w_match: {
      si val = eval(env, hd(a));
      a = tl(a);
      vec records;
      vinit(&records);
      for (;;) {
        si p = hd(a);
        a = tl(a);
        if (a == nil) {
          a = eval(env, p);
          break;
        }
        si r = hd(a);
        a = tl(a);
        records.n = 0;
        if (match(p, val, &records)) {
          for (si i = 0; i < records.n; i++)
            env = cons(records.p[i], env);
          a = eval(env, r);
          break;
        }
      }
      vfree(&records);
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

si evals(si env, si s0) {
  assert(tag(s0) == t_cons);

  // vars
  for (si s = s0; s != nil; s = tl(s)) {
    si a = hd(s);
    if (hd(a) != mkeyword(w_var))
      continue;
    a = tl(a);
    si key = hd(a);
    a = tl(a);
    si val = eval(env, hd(a));
    env = cons(list3(mkeyword(w_let), key, val), env);
  }

  // fns
  vec record;
  vinit(&record);
  for (si s = s0; s != nil; s = tl(s)) {
    si a = hd(s);
    if (hd(a) != mkeyword(w_fn))
      continue;
    a = tl(a);
    vpush(&record, hd(a));
    a = tl(a);
    vpush(&record, hd(a));
    a = tl(a);
    vpush(&record, cons(mkeyword(w_do), a));
  }
  env = cons(cons(mkeyword(w_letrec), list(&record)), env);

  // result
  si r = nil;
  for (si s = s0; s != nil; s = tl(s)) {
    si a = hd(s);
    si a0 = a;
    switch (keyword(hd(a))) {
    case s_assert_not:
      a = tl(a);
      if (!istrue(eval(env, hd(a))))
        continue;
      print(stderr, a0);
      err("assert-not: failed");
    case s_asserteq: {
      a = tl(a);
      si x = eval(env, hd(a));
      si y = eval(env, hd(tl(a)));
      if (x == y)
        continue;
      print(stderr, a0);
      print(stderr, x);
      print(stderr, y);
      err("assert=: failed");
    }
    case w_assert:
      a = tl(a);
      if (istrue(eval(env, hd(a))))
        continue;
      print(stderr, a0);
      err("assert: failed");
    case w_fn:
    case w_var:
      continue;
    case w_prin:
      a = tl(a);
      prin(stdout, eval(env, hd(a)));
      continue;
    case w_print:
      a = tl(a);
      print(stdout, eval(env, hd(a)));
      continue;
    }
    r = eval(env, a);
  }
  return r;
}
