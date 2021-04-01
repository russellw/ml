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
#define debug(a) printf("%s:%d: %s: %zx\n", __FILE__, __LINE__, #a, a);

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

#endif

#define isalpha1(c) (islower1(c) || isupper1(c))
#define isdigit1(c) ('0' <= (c) && (c) <= '9')
#define islower1(c) ('a' <= (c) && (c) <= 'z')
#define ispow2(n) (!((n) & (n)-1))
#define isspace1(c) ((c) <= ' ' && (c))
#define isupper1(c) ('A' <= (c) && (c) <= 'Z')

// SORT
extern char buf[20000];
extern jmp_buf jmpbuf;
///

// SORT
char *basename(char *file);
noret err(char *msg);
size_t fnv(char *s, si n);
void *mmalloc(si n);
void quote(char q, char *s);
void *xcalloc(si n, si size);
void *xmalloc(si n);
void *xrealloc(void *p, si n);
///
