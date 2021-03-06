#ifdef _MSC_VER
#define noret __declspec(noreturn) void
#else
#define noret __attribute__((__noreturn__)) void
#endif

typedef intptr_t si;

#ifdef DEBUG

void stacktrace();
si assertfail(char *file, si line, char *s);
#define assert(a) (a) || assertfail(__FILE__, __LINE__, #a)
#define unreachable assert(0)
#define debug(a)                                                               \
  fprintf(stderr, "%s:%d: %s: %zx\n", __FILE__, __LINE__, #a, (si)(a));
#define debugt(a)                                                              \
  do {                                                                         \
    fprintf(stderr, "%s:%d: %s: ", __FILE__, __LINE__, #a);                    \
    print(stderr, a);                                                          \
  } while (0);

#else

#define stacktrace()
#ifdef _MSC_VER
#define assert(a) __assume(a)
#define unreachable __assume(0)
#else
#define assert(a)
#define unreachable __builtin_unreachable()
#endif
#define debug(a)
#define debugt(a)

#endif

// the ctype.h versions of these contain extra logic that is not necessary or
// desirable for our purposes here, so define simple ones that do all and only
// what is needed
#define isalpha1(c) (islower1(c) || isupper1(c))
#define isdigit1(c) ('0' <= (c) && (c) <= '9')
#define islower1(c) ('a' <= (c) && (c) <= 'z')
#define isspace1(c) ((c) <= ' ' && (c))
#define isupper1(c) ('A' <= (c) && (c) <= 'Z')
#define isxdigit1(c) (xdigit(c) >= 0)

#define ispow2(n) (!((n) & (n)-1))

// SORT
extern char buf[20000];
extern jmp_buf jmpbuf;
///

// SORT
char *basename(char *file);
size_t fnv(char *s, si n);
void *mmalloc(si n);
void *xcalloc(si n, si size);
int xdigit(int c);
void *xmalloc(si n);
void *xrealloc(void *p, si n);
///
