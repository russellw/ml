typedef struct {
  mpq_t val;
} Rat;

void init_rats(void);
Rat *intern_rat(Rat *x);
