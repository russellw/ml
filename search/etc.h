#ifdef _MSC_VER
#define noret __declspec(noreturn) void
#else
#define noret __attribute__((__noreturn__)) void
#endif

typedef intptr_t si;

// SORT
inline void print(int32_t a) { printf("%d", a); }
inline void print(int64_t a) { printf("%lld", a); }
inline void print(uint32_t a) { printf("%u", a); }
inline void print(uint64_t a) { printf("%llu", a); }
inline void print(void *a) { printf("%p", a); }
///

template <class T, class U> void print(const pair<T, U> &p) {
  putchar('(');
  print(p.first);
  putchar(',');
  print(p.second);
  putchar(')');
}

#ifdef DEBUG

void stackTrace();
bool assertFail(const char *file, si line, const char *s);
#define assert(a) (a) || assertFail(__FILE__, __LINE__, #a)
#define unreachable assert(0)
#define debug(a)                                                               \
  do {                                                                         \
    printf("%s:%d: %s: ", __FILE__, __LINE__, #a);                             \
    print(a);                                                                  \
    putchar('\n');                                                             \
  } while (0)

#else

#define stackTrace()
#ifdef _MSC_VER
#define assert(a) __assume(a)
#define unreachable __assume(0)
#else
#define assert(a)
#define unreachable __builtin_unreachable()
#endif
#define debug(a)

#endif

extern char buf[20000];

// SORT
inline bool isDigit(char c) { return '0' <= c && c <= '9'; }

inline bool isLower(char c) { return 'a' <= c && c <= 'z'; }

inline bool isPow2(si n) {
  // doesn't work for 0
  assert(n);
  return !(n & n - 1);
}

inline bool isSpace(char c) { return c <= ' ' && c; }

inline bool isUpper(char c) { return 'A' <= c && c <= 'Z'; }
///

inline bool isAlpha(char c) { return isLower(c) || isUpper(c); }

// SORT
const char *basename(const char *file);
noret err(const char *msg);
size_t fnv(const void *p, si n);
void *mmalloc(si n);
void quote(char q, const char *s);
void *xcalloc(si n, si size);
void *xmalloc(si n);
void *xrealloc(void *p, si n);
///

#ifdef DEBUG
void ckPtr(const void *p);
void ckStr(const char *s);
#else
inline void ckPtr(const void *p) {}
inline void ckStr(const char *s) {}
#endif
