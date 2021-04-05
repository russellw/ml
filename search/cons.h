typedef struct {
  // unlike classic Lisp, tail must be a (possibly empty) list
  si hd, tl;
} Cons;

void init_cons(void);

// SORT
si cons(si hd, si tl);
si get(si env, si key, int *found);
si hd(si s);
si list(vec *v);
si list1(si a);
si list2(si a, si b);
si list3(si a, si b, si c);
si list4(si a, si b, si c, si d);
si tl(si s);
///
