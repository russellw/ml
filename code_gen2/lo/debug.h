#ifdef DEBUG

// Print stack trace. Intended for use in assert failure, but can also be used separately. Currently only implemented on Windows.
void stackTrace();

// Assert; unlike the standard library one, this one prints a stack trace.
[[noreturn]] bool assertFail(const char* file, int line, const char* func, const char* s);
#define assert(a) (a) || assertFail(__FILE__, __LINE__, __func__, #a)
#define unreachable assert(0)

#else

#define stackTrace()
#ifdef _MSC_VER
#define assert(a) __assume(a)
#define unreachable __assume(0)
#else
#define assert(a)
#define unreachable __builtin_unreachable()
#endif

#endif
