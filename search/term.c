#include "main.h"

// SORT
si abs1(si a) {
  switch (tag(a)) {
  case t_float:
    return ifloat(fabs(floatp(a)->val));
  case t_int: {
    Int r;
    mpz_init(r.val);
    mpz_abs(r.val, intp(a)->val);
    return iint(&r);
  }
  case t_rat: {
    Rat r;
    mpq_init(r.val);
    mpq_abs(r.val, ratp(a)->val);
    return irat(&r);
  }
  }
  err("abs: not a number");
  return 0;
}

si add(si a, si b) {
  switch (tag(a)) {
  case t_float:
    switch (tag(b)) {
    case t_float:
      return ifloat(floatp(a)->val + floatp(b)->val);
    case t_int:
      return ifloat(floatp(a)->val + mpz_get_d(intp(b)->val));
    case t_rat:
      return ifloat(floatp(a)->val + mpq_get_d(ratp(b)->val));
    }
    break;
  case t_int:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpz_get_d(intp(a)->val) + floatp(b)->val);
    case t_int: {
      Int r;
      mpz_init(r.val);
      mpz_add(r.val, intp(a)->val, intp(b)->val);
      return iint(&r);
    }
    case t_rat: {
      mpq_t a1;
      mpq_init(a1);
      mpq_set_z(a1, intp(a)->val);
      Rat r;
      mpq_init(r.val);
      mpq_add(r.val, a1, ratp(b)->val);
      mpq_clear(a1);
      return irat(&r);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpq_get_d(ratp(a)->val) + floatp(b)->val);
    case t_int: {
      mpq_t b1;
      mpq_init(b1);
      mpq_set_z(b1, intp(b)->val);
      Rat r;
      mpq_init(r.val);
      mpq_add(r.val, ratp(a)->val, b1);
      mpq_clear(b1);
      return irat(&r);
    }
    case t_rat: {
      Rat r;
      mpq_init(r.val);
      mpq_add(r.val, ratp(a)->val, ratp(b)->val);
      return irat(&r);
    }
    }
    break;
  }
  err("+: not a number");
  return 0;
}

si at(si s, si i) {
  if (tag(s) != t_cons)
    err("at: not a list");
  return 0;
}

si ceil1(si a) {
  switch (tag(a)) {
  case t_float:
    return ifloat(ceil(floatp(a)->val));
  case t_int:
    return a;
  case t_rat: {
    Int r;
    mpz_init(r.val);
    mpz_cdiv_q(r.val, mpq_numref(ratp(a)->val), mpq_denref(ratp(a)->val));
    return iint(&r);
  }
  }
  err("ceil: not a number");
  return 0;
}

si div2(si a, si b) {
  switch (tag(a)) {
  case t_float:
    switch (tag(b)) {
    case t_float:
      return ifloat(floatp(a)->val / floatp(b)->val);
    case t_int:
      return ifloat(floatp(a)->val / mpz_get_d(intp(b)->val));
    case t_rat:
      return ifloat(floatp(a)->val / mpq_get_d(ratp(b)->val));
    }
    break;
  case t_int:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpz_get_d(intp(a)->val) / floatp(b)->val);
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      Rat r;
      mpq_init(r.val);
      mpz_set(mpq_numref(r.val), intp(a)->val);
      mpz_set(mpq_denref(r.val), intp(b)->val);
      return irat(&r);
    }
    case t_rat: {
      // if the second operand were zero, it would already have been reduced to
      // an integer
      assert(mpq_sgn(ratp(b)->val));
      mpq_t a1;
      mpq_init(a1);
      mpq_set_z(a1, intp(a)->val);
      Rat r;
      mpq_init(r.val);
      mpq_div(r.val, a1, ratp(b)->val);
      mpq_clear(a1);
      return irat(&r);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpq_get_d(ratp(a)->val) / floatp(b)->val);
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      mpq_t b1;
      mpq_init(b1);
      mpq_set_z(b1, intp(b)->val);
      Rat r;
      mpq_init(r.val);
      mpq_div(r.val, ratp(a)->val, b1);
      mpq_clear(b1);
      return irat(&r);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));
      Rat r;
      mpq_init(r.val);
      mpq_div(r.val, ratp(a)->val, ratp(b)->val);
      return irat(&r);
    }
    }
    break;
  }
  err("/: not a number");
  return 0;
}

si div_c(si a, si b) {
  switch (tag(a)) {
  case t_int:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      Int q;
      mpz_init(q.val);
      mpz_cdiv_q(q.val, intp(a)->val, intp(b)->val);
      return iint(&q);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));
      Int q;
      mpz_init(q.val);
      mpz_mul(q.val, intp(a)->val, mpq_denref(ratp(b)->val));
      mpz_cdiv_q(q.val, q.val, mpq_numref(ratp(b)->val));
      return iint(&q);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");

      mpz_t nden_d;
      mpz_init(nden_d);
      mpz_mul(nden_d, mpq_denref(ratp(a)->val), intp(b)->val);

      Int q;
      mpz_init(q.val);
      mpz_cdiv_q(q.val, mpq_numref(ratp(a)->val), nden_d);

      mpz_clear(nden_d);
      return iint(&q);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));

      mpz_t nnum_dden;
      mpz_init(nnum_dden);
      mpz_mul(nnum_dden, mpq_numref(ratp(a)->val), mpq_denref(ratp(b)->val));

      mpz_t nden_dnum;
      mpz_init(nden_dnum);
      mpz_mul(nden_dnum, mpq_denref(ratp(a)->val), mpq_numref(ratp(b)->val));

      Int q;
      mpz_init(q.val);
      mpz_cdiv_q(q.val, nnum_dden, nden_dnum);

      mpz_clear(nden_dnum);
      mpz_clear(nnum_dden);
      return iint(&q);
    }
    }
    break;
  }
  err("div_c: not an exact number");
  return 0;
}

si div_e(si a, si b) {
  switch (tag(a)) {
  case t_int:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      Int q;
      mpz_init(q.val);
      mpz_ediv_q(q.val, intp(a)->val, intp(b)->val);
      return iint(&q);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));
      Int q;
      mpz_init(q.val);
      mpz_mul(q.val, intp(a)->val, mpq_denref(ratp(b)->val));
      mpz_ediv_q(q.val, q.val, mpq_numref(ratp(b)->val));
      return iint(&q);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");

      mpz_t nden_d;
      mpz_init(nden_d);
      mpz_mul(nden_d, mpq_denref(ratp(a)->val), intp(b)->val);

      Int q;
      mpz_init(q.val);
      mpz_ediv_q(q.val, mpq_numref(ratp(a)->val), nden_d);

      mpz_clear(nden_d);
      return iint(&q);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));

      mpz_t nnum_dden;
      mpz_init(nnum_dden);
      mpz_mul(nnum_dden, mpq_numref(ratp(a)->val), mpq_denref(ratp(b)->val));

      mpz_t nden_dnum;
      mpz_init(nden_dnum);
      mpz_mul(nden_dnum, mpq_denref(ratp(a)->val), mpq_numref(ratp(b)->val));

      Int q;
      mpz_init(q.val);
      mpz_ediv_q(q.val, nnum_dden, nden_dnum);

      mpz_clear(nden_dnum);
      mpz_clear(nnum_dden);
      return iint(&q);
    }
    }
    break;
  }
  err("div_e: not an exact number");
  return 0;
}

si div_f(si a, si b) {
  switch (tag(a)) {
  case t_int:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      Int q;
      mpz_init(q.val);
      mpz_fdiv_q(q.val, intp(a)->val, intp(b)->val);
      return iint(&q);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));
      Int q;
      mpz_init(q.val);
      mpz_mul(q.val, intp(a)->val, mpq_denref(ratp(b)->val));
      mpz_fdiv_q(q.val, q.val, mpq_numref(ratp(b)->val));
      return iint(&q);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");

      mpz_t nden_d;
      mpz_init(nden_d);
      mpz_mul(nden_d, mpq_denref(ratp(a)->val), intp(b)->val);

      Int q;
      mpz_init(q.val);
      mpz_fdiv_q(q.val, mpq_numref(ratp(a)->val), nden_d);

      mpz_clear(nden_d);
      return iint(&q);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));

      mpz_t nnum_dden;
      mpz_init(nnum_dden);
      mpz_mul(nnum_dden, mpq_numref(ratp(a)->val), mpq_denref(ratp(b)->val));

      mpz_t nden_dnum;
      mpz_init(nden_dnum);
      mpz_mul(nden_dnum, mpq_denref(ratp(a)->val), mpq_numref(ratp(b)->val));

      Int q;
      mpz_init(q.val);
      mpz_fdiv_q(q.val, nnum_dden, nden_dnum);

      mpz_clear(nden_dnum);
      mpz_clear(nnum_dden);
      return iint(&q);
    }
    }
    break;
  }
  err("div_f: not an exact number");
  return 0;
}

si div_t2(si a, si b) {
  switch (tag(a)) {
  case t_int:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      Int q;
      mpz_init(q.val);
      mpz_tdiv_q(q.val, intp(a)->val, intp(b)->val);
      return iint(&q);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));
      Int q;
      mpz_init(q.val);
      mpz_mul(q.val, intp(a)->val, mpq_denref(ratp(b)->val));
      mpz_tdiv_q(q.val, q.val, mpq_numref(ratp(b)->val));
      return iint(&q);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");

      mpz_t nden_d;
      mpz_init(nden_d);
      mpz_mul(nden_d, mpq_denref(ratp(a)->val), intp(b)->val);

      Int q;
      mpz_init(q.val);
      mpz_tdiv_q(q.val, mpq_numref(ratp(a)->val), nden_d);

      mpz_clear(nden_d);
      return iint(&q);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));

      mpz_t nnum_dden;
      mpz_init(nnum_dden);
      mpz_mul(nnum_dden, mpq_numref(ratp(a)->val), mpq_denref(ratp(b)->val));

      mpz_t nden_dnum;
      mpz_init(nden_dnum);
      mpz_mul(nden_dnum, mpq_denref(ratp(a)->val), mpq_numref(ratp(b)->val));

      Int q;
      mpz_init(q.val);
      mpz_tdiv_q(q.val, nnum_dden, nden_dnum);

      mpz_clear(nden_dnum);
      mpz_clear(nnum_dden);
      return iint(&q);
    }
    }
    break;
  }
  err("div_t: not an exact number");
  return 0;
}

si exact(si a) {
  switch (tag(a)) {
  case t_float: {
    Rat r;
    mpq_init(r.val);
    mpq_set_d(r.val, floatp(a)->val);
    return irat(&r);
  }
  case t_int:
  case t_rat:
    return a;
  }
  err("exact: not a number");
  return 0;
}

si floor1(si a) {
  switch (tag(a)) {
  case t_float:
    return ifloat(floor(floatp(a)->val));
  case t_int:
    return a;
  case t_rat: {
    Int r;
    mpz_init(r.val);
    mpz_fdiv_q(r.val, mpq_numref(ratp(a)->val), mpq_denref(ratp(a)->val));
    return iint(&r);
  }
  }
  err("floor: not a number");
  return 0;
}

si hd(si s) {
  if (tag(s) != t_cons)
    err("hd: not a list");
  if (s == empty)
    err("hd: empty list");
  return consp(s)->hd;
}

si inexact(si a) {
  switch (tag(a)) {
  case t_float:
    return a;
  case t_int:
    return ifloat(mpz_get_d(intp(a)->val));
  case t_rat:
    return ifloat(mpq_get_d(ratp(a)->val));
  }
  err("inexact: not a number");
  return 0;
}

int istrue(si a) {
  switch (tag(a)) {
  case t_float:
    return floatp(a)->val != 0;
  case t_int:
    return mpz_sgn(intp(a)->val);
  case t_rat:
    return mpq_sgn(ratp(a)->val);
  }
  return 1;
}

int le(si a, si b) {
  switch (tag(a)) {
  case t_float:
    switch (tag(b)) {
    case t_float:
      return floatp(a)->val <= floatp(b)->val;
    case t_int:
      return floatp(a)->val <= mpz_get_d(intp(b)->val);
    case t_rat:
      return floatp(a)->val <= mpq_get_d(ratp(b)->val);
    }
    break;
  case t_int:
    switch (tag(b)) {
    case t_float:
      return mpz_get_d(intp(a)->val) <= floatp(b)->val;
    case t_int:
      return mpz_cmp(intp(a)->val, intp(b)->val) <= 0;
    case t_rat: {
      mpq_t a1;
      mpq_init(a1);
      mpq_set_z(a1, intp(a)->val);
      int c = mpq_cmp(a1, ratp(b)->val);
      mpq_clear(a1);
      return c <= 0;
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_float:
      return mpq_get_d(ratp(a)->val) <= floatp(b)->val;
    case t_int: {
      mpq_t b1;
      mpq_init(b1);
      mpq_set_z(b1, intp(b)->val);
      int c = mpq_cmp(ratp(a)->val, b1);
      mpq_clear(b1);
      return c <= 0;
    }
    case t_rat:
      return mpq_cmp(ratp(a)->val, ratp(b)->val) <= 0;
    }
    break;
  }
  err("<=: not a number");
  return 0;
}

int lt(si a, si b) {
  switch (tag(a)) {
  case t_float:
    switch (tag(b)) {
    case t_float:
      return floatp(a)->val < floatp(b)->val;
    case t_int:
      return floatp(a)->val < mpz_get_d(intp(b)->val);
    case t_rat:
      return floatp(a)->val < mpq_get_d(ratp(b)->val);
    }
    break;
  case t_int:
    switch (tag(b)) {
    case t_float:
      return mpz_get_d(intp(a)->val) < floatp(b)->val;
    case t_int:
      return mpz_cmp(intp(a)->val, intp(b)->val) < 0;
    case t_rat: {
      mpq_t a1;
      mpq_init(a1);
      mpq_set_z(a1, intp(a)->val);
      int c = mpq_cmp(a1, ratp(b)->val);
      mpq_clear(a1);
      return c < 0;
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_float:
      return mpq_get_d(ratp(a)->val) < floatp(b)->val;
    case t_int: {
      mpq_t b1;
      mpq_init(b1);
      mpq_set_z(b1, intp(b)->val);
      int c = mpq_cmp(ratp(a)->val, b1);
      mpq_clear(b1);
      return c < 0;
    }
    case t_rat:
      return mpq_cmp(ratp(a)->val, ratp(b)->val) < 0;
    }
    break;
  }
  err("<: not a number");
  return 0;
}

si minus(si a) {
  switch (tag(a)) {
  case t_float:
    return ifloat(-floatp(a)->val);
  case t_int: {
    Int r;
    mpz_init(r.val);
    mpz_neg(r.val, intp(a)->val);
    return iint(&r);
  }
  case t_rat: {
    Rat r;
    mpq_init(r.val);
    mpq_neg(r.val, ratp(a)->val);
    return irat(&r);
  }
  }
  err("minus: not a number");
  return 0;
}

si mul(si a, si b) {
  switch (tag(a)) {
  case t_float:
    switch (tag(b)) {
    case t_float:
      return ifloat(floatp(a)->val * floatp(b)->val);
    case t_int:
      return ifloat(floatp(a)->val * mpz_get_d(intp(b)->val));
    case t_rat:
      return ifloat(floatp(a)->val * mpq_get_d(ratp(b)->val));
    }
    break;
  case t_int:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpz_get_d(intp(a)->val) * floatp(b)->val);
    case t_int: {
      Int r;
      mpz_init(r.val);
      mpz_mul(r.val, intp(a)->val, intp(b)->val);
      return iint(&r);
    }
    case t_rat: {
      mpq_t a1;
      mpq_init(a1);
      mpq_set_z(a1, intp(a)->val);
      Rat r;
      mpq_init(r.val);
      mpq_mul(r.val, a1, ratp(b)->val);
      mpq_clear(a1);
      return irat(&r);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpq_get_d(ratp(a)->val) * floatp(b)->val);
    case t_int: {
      mpq_t b1;
      mpq_init(b1);
      mpq_set_z(b1, intp(b)->val);
      Rat r;
      mpq_init(r.val);
      mpq_mul(r.val, ratp(a)->val, b1);
      mpq_clear(b1);
      return irat(&r);
    }
    case t_rat: {
      Rat r;
      mpq_init(r.val);
      mpq_mul(r.val, ratp(a)->val, ratp(b)->val);
      return irat(&r);
    }
    }
    break;
  }
  err("*: not a number");
  return 0;
}

void print(si a) {
  switch (tag(a)) {
  case t_cons: {
    putchar('(');
    int more = 0;
    while (a != empty) {
      if (more)
        putchar(' ');
      more = 1;
      print(hd(a));
      a = tl(a);
    }
    putchar(')');
    return;
  }
  case t_float:
    printf("%f", floatp(a)->val);
    return;
  case t_int:
    mpz_out_str(stdout, 10, intp(a)->val);
    return;
  case t_rat:
    mpq_out_str(stdout, 10, ratp(a)->val);
    return;
  case t_sym:
    fwrite(symp(a)->v, symp(a)->n, 1, stdout);
    return;
  }
  unreachable;
}

void println(si a) {
  print(a);
  putchar('\n');
}

si rem_c(si a, si b) {
  switch (tag(a)) {
  case t_int:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      Int r;
      mpz_init(r.val);
      mpz_cdiv_r(r.val, intp(a)->val, intp(b)->val);
      return iint(&r);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));
      Int r;
      mpz_init(r.val);
      mpz_mul(r.val, intp(a)->val, mpq_denref(ratp(b)->val));
      mpz_cdiv_r(r.val, r.val, mpq_numref(ratp(b)->val));
      return iint(&r);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");

      mpz_t nden_d;
      mpz_init(nden_d);
      mpz_mul(nden_d, mpq_denref(ratp(a)->val), intp(b)->val);

      Int r;
      mpz_init(r.val);
      mpz_cdiv_r(r.val, mpq_numref(ratp(a)->val), nden_d);

      mpz_clear(nden_d);
      return iint(&r);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));

      mpz_t nnum_dden;
      mpz_init(nnum_dden);
      mpz_mul(nnum_dden, mpq_numref(ratp(a)->val), mpq_denref(ratp(b)->val));

      mpz_t nden_dnum;
      mpz_init(nden_dnum);
      mpz_mul(nden_dnum, mpq_denref(ratp(a)->val), mpq_numref(ratp(b)->val));

      Int r;
      mpz_init(r.val);
      mpz_cdiv_r(r.val, nnum_dden, nden_dnum);

      mpz_clear(nden_dnum);
      mpz_clear(nnum_dden);
      return iint(&r);
    }
    }
    break;
  }
  err("rem_c: not an exact number");
  return 0;
}

si rem_e(si a, si b) {
  switch (tag(a)) {
  case t_int:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      Int r;
      mpz_init(r.val);
      mpz_ediv_r(r.val, intp(a)->val, intp(b)->val);
      return iint(&r);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));
      Int r;
      mpz_init(r.val);
      mpz_mul(r.val, intp(a)->val, mpq_denref(ratp(b)->val));
      mpz_ediv_r(r.val, r.val, mpq_numref(ratp(b)->val));
      return iint(&r);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");

      mpz_t nden_d;
      mpz_init(nden_d);
      mpz_mul(nden_d, mpq_denref(ratp(a)->val), intp(b)->val);

      Int r;
      mpz_init(r.val);
      mpz_ediv_r(r.val, mpq_numref(ratp(a)->val), nden_d);

      mpz_clear(nden_d);
      return iint(&r);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));

      mpz_t nnum_dden;
      mpz_init(nnum_dden);
      mpz_mul(nnum_dden, mpq_numref(ratp(a)->val), mpq_denref(ratp(b)->val));

      mpz_t nden_dnum;
      mpz_init(nden_dnum);
      mpz_mul(nden_dnum, mpq_denref(ratp(a)->val), mpq_numref(ratp(b)->val));

      Int r;
      mpz_init(r.val);
      mpz_ediv_r(r.val, nnum_dden, nden_dnum);

      mpz_clear(nden_dnum);
      mpz_clear(nnum_dden);
      return iint(&r);
    }
    }
    break;
  }
  err("rem_e: not an exact number");
  return 0;
}

si rem_f(si a, si b) {
  switch (tag(a)) {
  case t_int:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      Int r;
      mpz_init(r.val);
      mpz_fdiv_r(r.val, intp(a)->val, intp(b)->val);
      return iint(&r);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));
      Int r;
      mpz_init(r.val);
      mpz_mul(r.val, intp(a)->val, mpq_denref(ratp(b)->val));
      mpz_fdiv_r(r.val, r.val, mpq_numref(ratp(b)->val));
      return iint(&r);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");

      mpz_t nden_d;
      mpz_init(nden_d);
      mpz_mul(nden_d, mpq_denref(ratp(a)->val), intp(b)->val);

      Int r;
      mpz_init(r.val);
      mpz_fdiv_r(r.val, mpq_numref(ratp(a)->val), nden_d);

      mpz_clear(nden_d);
      return iint(&r);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));

      mpz_t nnum_dden;
      mpz_init(nnum_dden);
      mpz_mul(nnum_dden, mpq_numref(ratp(a)->val), mpq_denref(ratp(b)->val));

      mpz_t nden_dnum;
      mpz_init(nden_dnum);
      mpz_mul(nden_dnum, mpq_denref(ratp(a)->val), mpq_numref(ratp(b)->val));

      Int r;
      mpz_init(r.val);
      mpz_fdiv_r(r.val, nnum_dden, nden_dnum);

      mpz_clear(nden_dnum);
      mpz_clear(nnum_dden);
      return iint(&r);
    }
    }
    break;
  }
  err("rem_f: not an exact number");
  return 0;
}

si rem_t(si a, si b) {
  switch (tag(a)) {
  case t_int:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");
      Int r;
      mpz_init(r.val);
      mpz_tdiv_r(r.val, intp(a)->val, intp(b)->val);
      return iint(&r);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));
      Int r;
      mpz_init(r.val);
      mpz_mul(r.val, intp(a)->val, mpq_denref(ratp(b)->val));
      mpz_tdiv_r(r.val, r.val, mpq_numref(ratp(b)->val));
      return iint(&r);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_int: {
      if (!mpz_sgn(intp(b)->val))
        err("division by zero");

      mpz_t nden_d;
      mpz_init(nden_d);
      mpz_mul(nden_d, mpq_denref(ratp(a)->val), intp(b)->val);

      Int r;
      mpz_init(r.val);
      mpz_tdiv_r(r.val, mpq_numref(ratp(a)->val), nden_d);

      mpz_clear(nden_d);
      return iint(&r);
    }
    case t_rat: {
      assert(mpq_sgn(ratp(b)->val));

      mpz_t nnum_dden;
      mpz_init(nnum_dden);
      mpz_mul(nnum_dden, mpq_numref(ratp(a)->val), mpq_denref(ratp(b)->val));

      mpz_t nden_dnum;
      mpz_init(nden_dnum);
      mpz_mul(nden_dnum, mpq_denref(ratp(a)->val), mpq_numref(ratp(b)->val));

      Int r;
      mpz_init(r.val);
      mpz_tdiv_r(r.val, nnum_dden, nden_dnum);

      mpz_clear(nden_dnum);
      mpz_clear(nnum_dden);
      return iint(&r);
    }
    }
    break;
  }
  err("rem_t: not an exact number");
  return 0;
}

si round1(si a) {
  switch (tag(a)) {
  case t_float:
    return ifloat(round(floatp(a)->val));
  case t_int:
    return a;
  case t_rat: {
    Int r;
    mpz_init(r.val);
    mpz_qround(r.val, mpq_numref(ratp(a)->val), mpq_denref(ratp(a)->val));
    return iint(&r);
  }
  }
  err("round: not a number");
  return 0;
}

si sub(si a, si b) {
  switch (tag(a)) {
  case t_float:
    switch (tag(b)) {
    case t_float:
      return ifloat(floatp(a)->val - floatp(b)->val);
    case t_int:
      return ifloat(floatp(a)->val - mpz_get_d(intp(b)->val));
    case t_rat:
      return ifloat(floatp(a)->val - mpq_get_d(ratp(b)->val));
    }
    break;
  case t_int:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpz_get_d(intp(a)->val) - floatp(b)->val);
    case t_int: {
      Int r;
      mpz_init(r.val);
      mpz_sub(r.val, intp(a)->val, intp(b)->val);
      return iint(&r);
    }
    case t_rat: {
      mpq_t a1;
      mpq_init(a1);
      mpq_set_z(a1, intp(a)->val);
      Rat r;
      mpq_init(r.val);
      mpq_sub(r.val, a1, ratp(b)->val);
      mpq_clear(a1);
      return irat(&r);
    }
    }
    break;
  case t_rat:
    switch (tag(b)) {
    case t_float:
      return ifloat(mpq_get_d(ratp(a)->val) - floatp(b)->val);
    case t_int: {
      mpq_t b1;
      mpq_init(b1);
      mpq_set_z(b1, intp(b)->val);
      Rat r;
      mpq_init(r.val);
      mpq_sub(r.val, ratp(a)->val, b1);
      mpq_clear(b1);
      return irat(&r);
    }
    case t_rat: {
      Rat r;
      mpq_init(r.val);
      mpq_sub(r.val, ratp(a)->val, ratp(b)->val);
      return irat(&r);
    }
    }
    break;
  }
  err("-: not a number");
  return 0;
}

si tl(si s) {
  if (tag(s) != t_cons)
    err("tl: not a list");
  if (s == empty)
    err("tl: empty list");
  return consp(s)->tl;
}

si trunc1(si a) {
  switch (tag(a)) {
  case t_float:
    return ifloat(trunc(floatp(a)->val));
  case t_int:
    return a;
  case t_rat: {
    Int r;
    mpz_init(r.val);
    mpz_tdiv_q(r.val, mpq_numref(ratp(a)->val), mpq_denref(ratp(a)->val));
    return iint(&r);
  }
  }
  err("trunc: not a number");
  return 0;
}
///
