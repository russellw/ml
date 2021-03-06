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

#define version "3"

enum class language {
  unknown,

  dimacs,
  tptp,
};

namespace {
struct lineParser : parser {
  lineParser(const char *file, vec<char *> &v) : parser(file) {
    auto s = text;
    while (*s) {
      auto t = strchr(s, '\n');
      auto s1 = t + 1;
      if (t != s && t[-1] == '\r')
        --t;
      auto n = t - s;
      auto r = (char *)mmalloc(n + 1);
      memcpy(r, s, n);
      r[n] = 0;
      v.push_back(r);
      s = s1;
    }
  }
};

// SORT
language lang;
time_t timeLimit;
vec<const char *> files;
///

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
  auto s = strrchr(file, '.');
  return s ? s + 1 : "";
}

unsigned long long optNum(si argc, char **argv, si &i, const char *optArg) {
  if (!optArg) {
    if (i + 1 >= argc) {
      fprintf(stderr, "%s: expected argument\n", argv[i]);
      exit(1);
    }
    optArg = argv[++i];
  }
  errno = 0;
  char *e = 0;
  auto r = strtoull(optArg, &e, 10);
  if (errno) {
    perror(optArg);
    exit(1);
  }
  if (e == optArg) {
    fprintf(stderr, "%s: expected integer\n", optArg);
    exit(1);
  }
  return r;
}

void parse(si argc, char **argv) {
  for (si i = 0; i != argc; ++i) {
    auto s = argv[i];

    // file
    if (!strcmp(s, "-")) {
      files.push_back("stdin");
      continue;
    }
    if (*s != '-') {
      if (!strcmp(ext(s), "lst")) {
        vec<char *> v;
        lineParser p(s, v);
        parse(v.n, v.p);
        continue;
      }
      files.push_back(s);
      continue;
    }

    // option
    while (*s == '-')
      ++s;

    // optArg
    auto t = s;
    while (isAlpha(*t))
      ++t;
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
    switch (keyword(intern(s))) {
    case k_T: {
      auto seconds = optNum(argc, argv, i, optArg);
#ifdef _WIN32
      HANDLE timer = 0;
      CreateTimerQueueTimer(&timer, 0, timeout, 0, (DWORD)(seconds * 1000), 0,
                            WT_EXECUTEINTIMERTHREAD);
#else
      alarm(seconds);
#endif
      break;
    }
    case k_V:
    case k_v:
    case k_version:
      printf("Ayane " version ", %zu-bit "
#ifdef DEBUG
             "debug"
#else
             "release"
#endif
             " build\n",
             sizeof(void *) * 8);
      exit(0);
    case k_dimacs:
      lang = language::dimacs;
      break;
    case k_h:
    case k_help:
      help();
      exit(0);
    case k_t:
      timeLimit = optNum(argc, argv, i, optArg);
      break;
    case k_tptp:
      lang = language::tptp;
      break;
    default:
      fprintf(stderr, "%s: unknown option\n", argv[i]);
      exit(1);
    }
  }
}

language getLang(const char *file) {
  if (lang != language::unknown)
    return lang;
  switch (keyword(intern(ext(file)))) {
  case k_cnf:
    return language::dimacs;
  }
  return language::tptp;
}

#if defined DEBUG && defined _WIN32
void printItem1(si n, const char *caption) {
  auto s = buf + sizeof buf - 1;
  *s = 0;
  si i = 0;
  do {
    // extract a digit
    *--s = '0' + n % 10;
    n /= 10;

    // track how many digits we have extracted
    ++i;

    // so that we can punctuate them in groups of 3
    if (i % 3 == 0 && n)
      *--s = ',';
  } while (n);
  si used = buf + sizeof buf - s;
  si spaces = 15 - used;
  for (i = 0; i < spaces; ++i)
    putchar(' ');
  printf("%s  %s\n", s, caption);
}

#define printItem(x) printItem1(pmc.x, #x)

void printMem() {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&pmc,
                       sizeof pmc);
  printItem(PageFaultCount);
  printItem(PeakWorkingSetSize);
  printItem(WorkingSetSize);
  printItem(QuotaPeakPagedPoolUsage);
  printItem(QuotaPagedPoolUsage);
  printItem(QuotaPeakNonPagedPoolUsage);
  printItem(QuotaNonPagedPoolUsage);
  printItem(PagefileUsage);
  printItem(PeakPagefileUsage);
  printItem(PrivateUsage);
}
#endif
} // namespace

#ifdef BENCHMARK
term fn(type t, sym *s) {
  if (s->t == type::none)
    s->t = t;
  assert(s->t == t);
  return tag(term::Sym, s);
}

term mkInt(int val) {
  Int x;
  mpz_init_set_si(x.val, val);
  return tag(term::Int, intern(x));
}

void benchmark() {
  auto start = GetTickCount();
  si iters = 0;
  unsigned r = 0;

  // init
  initSolver();
  vec<pair<term, term>> unified;
  auto addInt =
      fn(internType(type::Int, type::Int, type::Int), intern("addInt"));
  auto a = intern(term::Eq, intern(term::Call, addInt, mkInt(1), mkInt(2)),
                  intern(term::Call, addInt, mkInt(1), mkInt(2)));
  auto b = intern(term::Eq, var(type::Int, 0), var(type::Int, 1));

  while (GetTickCount() < start + 10000) {
    // avoid calling GetTickCount every iteration; it takes a couple of
    // nanoseconds per call, which for some operations might add significantly
    // to the measured time
    for (si i = 0; i != 1000; ++i) {
      unified.n = 0;
      r += unify(a, b, unified);
      ++iters;
    }
  }
  printf("ps / iter: %llu\n", (GetTickCount() - start) * 1000000000ULL / iters);
  // return a result to make sure the calculation doesn't get optimized away
  exit(r);
}
#else
#define benchmark()
#endif

int main(int argc, char **argv) {
  set_new_handler([]() {
    perror("new");
    exit(1);
  });

#ifdef DEBUG
  // fast tests always run in debug build
  test();
  assert(_CrtCheckMemory());
#if 1
  // slow tests only run when needed
  test1();
  assert(_CrtCheckMemory());
#endif
#endif

  benchmark();

  parse(argc - 1, argv + 1);
  if (!files.n) {
    if (isatty(0)) {
      help();
      return 0;
    }
    files.push_back("stdin");
  }
  assert(_CrtCheckMemory());

  for (si i = 0; i != files.n; ++i) {
    auto file = files[i];
    auto bname = basename(file);
    auto start = time(0);
    if (timeLimit)
      deadline = start + timeLimit;
    initSolver();
    assert(_CrtCheckMemory());
    try {
      switch (getLang(file)) {
      case language::dimacs:
        dimacs(file);
        break;
      case language::tptp:
        tptp(file);
        break;
      default:
        unreachable;
      }
      assert(_CrtCheckMemory());
      auto r = saturate();
      assert(_CrtCheckMemory());
      if (conjecture)
        switch (r) {
        case szs::Satisfiable:
          r = szs::CounterSatisfiable;
          break;
        case szs::Unsatisfiable:
          r = szs::Theorem;
          break;
        }
      printf("%% SZS status %s for %s\n", szsNames[(si)r], bname);
#ifdef DEBUG
      if (expected != szs::none && r != expected)
        switch (r) {
        case szs::CounterSatisfiable:
        case szs::Satisfiable:
          printf("error: expected %s\n", szsNames[(si)expected]);
          return 1;
        case szs::Theorem:
        case szs::Unsatisfiable:
          if (expected == szs::ContradictoryAxioms)
            break;
        }
#ifdef _WIN32
      putchar('\n');
      printMem();
      putchar('\n');
#endif
      printf("%zu seconds\n", (si)(time(0) - start));
#endif
    } catch (inappropriate e) {
      printf("%% SZS status Inappropriate for %s\n", bname);
    }
    if (i + 1 < files.n)
      putchar('\n');
  }
  return 0;
}
