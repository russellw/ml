void mpz_ediv_q(mpz_t q, mpz_t n, mpz_t d);
void mpz_ediv_r(mpz_t r, mpz_t n, mpz_t d);

void round(mpz_t q, mpz_t n, mpz_t d);

struct Int {
  mpz_t val;

  unsigned hash() const { return mpz_get_ui(val); }
  bool eq(const Int &x) const { return !mpz_cmp(val, x.val); }
  void clear() { mpz_clear(val); }
};

struct Rat {
  mpq_t val;

  unsigned hash() const {
    return mpz_get_ui(mpq_numref(val)) ^ mpz_get_ui(mpq_denref(val));
  }
  bool eq(const Rat &x) const { return mpq_equal(val, x.val); }
  void clear() { mpq_clear(val); }
};

Int *intern(Int &x);
Rat *intern(Rat &x);

void initNums();

#ifdef DEBUG
void ck(const Int *x);
void ck(const Rat *x);
#else
inline void ck(const Int *x) {}
inline void ck(const Rat *x) {}
#endif

void print(const Int &x);
void print(const Rat &x);
