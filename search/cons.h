typedef struct {
  // unlike classic Lisp, tail must be a (possibly empty) list
  si hd, tl;
} Cons;

void init_cons(void);

// SORT
si cons(si hd, si tl);
si hd(si s);
si tl(si s);
///
