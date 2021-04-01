typedef struct {
  // unlike classic Lisp, tail must be a (possibly empty) list
  si hd, tl;
} Cons;

void init_cons(void);
si cons(si hd, si tl);
