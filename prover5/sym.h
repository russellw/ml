struct sym {
  // type named by this symbol
  type thisType;

  // type of function named by this symbol
  type t;

  // for the keyword system to work, the size of the declared character array
  // must be large enough to hold the longest keyword

  // for the system to work efficiently, the size of the whole structure must be
  // a power of 2

  // when symbols are allocated on the heap, the code doing the allocation is
  // responsible for allocating enough space to hold the corresponding strings
  char v[0x20 - 2 * sizeof(type)];
};

inline si keyword(const sym *s) {
  // turn a symbol into a keyword number by subtracting the base of the keyword
  // array and dividing by the declared size of a symbol structure (which is
  // efficient as long as that size is a power of 2)

  // it's okay if the symbol is not a keyword; that just means the resulting
  // number will not correspond to any keyword and will not match any case in a
  // switch statement
  size_t i = (const char *)s - (const char *)keywords;
  return i / sizeof *s;
}

sym *mkSym(type t);
sym *intern(const char *s, si n);
inline sym *intern(const char *s) { return intern(s, strlen(s)); }

void initSyms();

#ifdef DEBUG
void ck(const sym *s);
#else
inline void ck(const sym *s) {}
#endif
