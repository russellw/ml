typedef struct {
  si n;

  // for the keyword system to work, the size of the declared character array
  // must be large enough to hold the longest keyword

  // for the system to work efficiently, the size of the whole structure must be
  // a power of 2

  // when symbols are allocated on the heap, the code doing the allocation is
  // responsible for allocating enough space to hold the corresponding strings
  char v[0x20 - sizeof(si)];
} sym;

void init_syms(void);
si intern(char *s, si n);
static si internz(char *s) { return intern(s, strlen(s)); }
