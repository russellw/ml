#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

#ifdef _WIN32
#include <io.h>
#include <windows.h>
// windows.h must be first
#include <psapi.h>
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

const char *ext(const char *file) {
  // don't care about a.b/c
  char *s = strrchr(file, '.');
  return s ? s + 1 : "";
}

void parse(si argc, char **argv) {
  for (si i = 0; i != argc; ++i) {
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
    const char *optArg = 0;
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
      s = intern(s, t - s)->v;
      optArg = t;
      break;
    case ':':
    case '=':
      *t = 0;
      optArg = t + 1;
      break;
    }

    // option
    switch (keyword(intern(s, strlen(s)))) {
    case k_V:
    case k_v:
    case k_version:
      printf("Aklo " version ", %zu-bit "
#ifdef DEBUG
             "debug"
#else
             "release"
#endif
             " build\n",
             sizeof(void *) * 8);
      exit(0);
    case k_h:
    case k_help:
      help();
      exit(0);
    default:
      fprintf(stderr, "%s: unknown option\n", argv[i]);
      exit(1);
    }
  }
}

int main(int argc, char **argv) {
#ifdef DEBUG
  test();
  assert(_CrtCheckMemory());
#endif
  return 0;
}
