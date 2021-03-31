void mpz_ediv_q(mpz_t q, mpz_t n, mpz_t d);
void mpz_ediv_r(mpz_t r, mpz_t n, mpz_t d);

void round(mpz_t q, mpz_t n, mpz_t d);

typedef struct {
  mpz_t val;
} Int;

void init_ints(void);
Int *intern_int(Int *x);
