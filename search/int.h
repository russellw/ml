void mpz_ediv_q(mpz_t q, mpz_t n, mpz_t d);
void mpz_ediv_r(mpz_t r, mpz_t n, mpz_t d);

void round(mpz_t q, mpz_t n, mpz_t d);

struct Int {
  mpz_t val;
};
