#include "main.h"

// SORT
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
///

si eval(si a) { return a; }
