#include "main.h"

#ifdef DEBUG
#ifdef _WIN32
#include <windows.h>
// windows.h must be first
#include <dbghelp.h>
#endif

void stacktrace() {
#ifdef _WIN32
  HANDLE process = GetCurrentProcess();
  SymInitialize(process, 0, 1);
  void *stack[100];
  int nframes =
      CaptureStackBackTrace(1, sizeof stack / sizeof *stack, stack, 0);
  char buf[sizeof(SYMBOL_INFO) + 256];
  SYMBOL_INFO *syminfo = (SYMBOL_INFO *)buf;
  syminfo->MaxNameLen = 256;
  syminfo->SizeOfStruct = sizeof(SYMBOL_INFO);
  IMAGEHLP_LINE64 location;
  location.SizeOfStruct = sizeof location;
  for (int i = 0; i < nframes; i++) {
    DWORD64 address = (DWORD64)(stack[i]);
    SymFromAddr(process, address, 0, syminfo);
    DWORD displacement;
    if (SymGetLineFromAddr64(process, address, &displacement, &location))
      fprintf(stderr, "%s:%lu: ", location.FileName, location.LineNumber);
    fprintf(stderr, "%s\n", syminfo->Name);
  }
#endif
}

si assertfail(char *file, si line, char *s) {
  printf("%s:%zu: assert failed: %s\n", file, line, s);
  stacktrace();
  exit(1);
  // keep the compiler happy about the use of || in the assert macro
  return 0;
}
#endif

// buf is sized for largest TPTP symbols
// SORT
char buf[20000];
jmp_buf jmpbuf;
///

// SORT
char *basename(char *file) {
  si i = strlen(file);
  while (i) {
    if (file[i - 1] == '/')
      return file + i;
#ifdef _WIN32
    if (file[i - 1] == '\\')
      return file + i;
#endif
    --i;
  }
  return file;
}

noret err(char *msg) {
  strcpy(buf, msg);
  longjmp(jmpbuf, 1);
}

size_t fnv(char *s, si n) {
  // Fowler-Noll-Vo-1a is slower than more sophisticated hash algorithms for
  // large chunks of data, but faster for tiny ones, so it still sees use
  size_t h = 2166136261u;
  while (n--) {
    h ^= *s++;
    h *= 16777619;
  }
  return h;
}

void *mmalloc(si n) {
  // monotonic malloc, for memory that does not need to be freed until the
  // program exits
  n = n + 7 & ~(si)7;
  static char *p;
  static char *e;
  assert(!((si)p & 7));
  assert(!((si)e & 7));
  if (e - p < n) {
    si chunk = max(n, 10000);
    p = xmalloc(chunk);
    e = p + chunk;
  }
  char *r = p;
#ifdef DEBUG
  memset(r, 0xcc, n);
#endif
  p += n;
  return r;
}

void *xcalloc(si n, si size) {
  void *r = calloc(n, size);
  if (!r) {
    perror("calloc");
    exit(1);
  }
  return r;
}

void *xmalloc(si n) {
  void *r = malloc(n);
  if (!r) {
    perror("malloc");
    exit(1);
  }
#ifdef DEBUG
  memset(r, 0xcc, n);
#endif
  return r;
}

void *xrealloc(void *p, si n) {
  void *r = realloc(p, n);
  if (!r) {
    perror("realloc");
    exit(1);
  }
  return r;
}

int xdigit(int c) {
  if (isdigit1(c))
    return c - '0';
  if ('a' <= c && c <= 'f')
    return c - 'a' + 10;
  if ('A' <= c && c <= 'F')
    return c - 'A' + 10;
  return -1;
}
///
