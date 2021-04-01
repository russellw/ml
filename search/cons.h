typedef struct {
  // unlike classic Lisp, cdr must be a (possibly empty) list
  si car, cdr;
} Cons;

void init_cons(void);
si cons(si car, si cdr);
