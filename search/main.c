#include "main.h"

#ifdef _WIN32
#include <io.h>
#include <windows.h>
// windows.h must be first
#include <psapi.h>

#ifdef DEBUG
static LONG WINAPI handler(struct _EXCEPTION_POINTERS *ExceptionInfo) {
  if (ExceptionInfo->ExceptionRecord->ExceptionCode ==
      EXCEPTION_STACK_OVERFLOW) {
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), "Stack overflow\n", 15, 0, 0);
    ExitProcess(1);
  }
  fprintf(stderr, "Exception code %lx\n",
          ExceptionInfo->ExceptionRecord->ExceptionCode);
  stacktrace();
  ExitProcess(1);
}
#endif

static VOID CALLBACK timeout(PVOID a, BOOLEAN b) {
  // on Linux the exit code associated with hard timeout is 128+SIGALRM; there
  // is no particular reason why we have to use the same exit code on Windows,
  // but there is no reason not to, either; as a distinctive exit code for this
  // purpose, it serves as well as any
  ExitProcess(128 + 14);
}
#else
#include <unistd.h>
#endif

#define version "0"

void help() {
  printf("General options:\n"
         "-help       show help\n"
         "-version    show version\n"
         "\n"
         "Input:\n"
         "-dimacs     dimacs format\n"
         "-tptp       tptp format\n"
         "-           read stdin\n"
         "\n"
         "Resources:\n"
         "-t seconds  soft time limit\n"
         "-T seconds  hard time limit\n");
}

char *ext(char *file) {
  // don't care about a.b/c
  char *s = strrchr(file, '.');
  return s ? s + 1 : "";
}

void parseargv(int argc, char **argv) {
  for (int i = 0; i != argc; ++i) {
    char *s = argv[i];

    // file
    if (!strcmp(s, "-")) {
      continue;
    }
    if (*s != '-') {
      continue;
    }

    // option
    while (*s == '-')
      s++;

    // optArg
    char *t = s;
    while (isalpha1(*t))
      t++;
    char *optArg = 0;
    switch (*t) {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      optArg = t;
      break;
    case ':':
    case '=':
      *t = 0;
      optArg = t + 1;
      break;
    }

    // option
    switch (keyword(internz(s))) {
    case w_V:
    case w_v:
    case w_version:
      printf("Aklo " version ", %zu-bit "
#ifdef DEBUG
             "debug"
#else
             "release"
#endif
             " build\n",
             sizeof(void *) * 8);
      exit(0);
    case w_h:
    case w_help:
      help();
      exit(0);
    default:
      fprintf(stderr, "%s: unknown option\n", argv[i]);
      exit(1);
    }
  }
}

int main(int argc, char **argv) {
#if defined(DEBUG) && defined(_WIN32)
  AddVectoredExceptionHandler(0, handler);
#endif

  // SORT
  init_cons();
  init_floats();
  init_ints();
  init_parser();
  init_rats();
  init_syms();
  ///

#ifdef DEBUG
  test();
  assert(_CrtCheckMemory());
#endif

  vec v;
  vinit(&v);
  parsefile("test.k", &v);
  parsefile("core.k", &v);
  for (si i = 0; i < v.n; i++) {
    si a = v.p[i];
    si op = hd(a);
    si a1 = tl(a);
    switch (keyword(op)) {
    case s_asserteq:
      if (eval(nil, hd(a1)) != eval(nil, hd(tl(a1)))) {
        println(a);
        puts("Assert failed");
        return 1;
      }
      break;
    case w_assert:
      if (!istrue(eval(nil, hd(a1)))) {
        println(a);
        puts("Assert failed");
        return 1;
      }
      break;
    }
  }
  return 0;
}
