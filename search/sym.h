static si keyword(sym *s) {
  // turn a symbol into a keyword number by subtracting the base of the keyword
  // array and dividing by the declared size of a symbol structure (which is
  // efficient as long as that size is a power of 2)

  // it's okay if the symbol is not a keyword; that just means the resulting
  // number will not correspond to any keyword and will not match any case in a
  // switch statement
  size_t i = (char *)s - (char *)keywords;
  return i / sizeof *s;
}

void init_syms(void);
sym *intern(char *s, si n);
static sym *internz(char *s) { return intern(s, strlen(s)); }
