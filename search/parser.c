#include "main.h"
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#define O_BINARY 0
#endif

// input file
char *file;

// input text
char *txtstart;
char *txt;

// input token
char *tokstart;
int bufi;
int tok;
si tokterm;

void readfile(void) {
  char *s = 0;
  si n = 0;
  if (strcmp(file, "stdin")) {
    // read from file
    int f = open(file, O_BINARY | O_RDONLY);
    struct stat st;
    if (f < 0 || fstat(f, &st)) {
      perror(file);
      exit(1);
    }

    // check size of file
    n = st.st_size;

    // allow space for null terminator
    s = xmalloc(n + 1);

    // read all the data
    if (read(f, s, n) != n) {
      perror("read");
      exit(1);
    }

    // and we are done with the file
    close(f);
  } else {
    // read from stdin
#ifdef _WIN32
    _setmode(0, O_BINARY);
#endif
    si chunk = 1 << 20;
    si cap = 0;

    // stdin doesn't tell us in advance how much data there will be, so keep
    // reading chunks until done
    for (;;) {
      // expand buffer as necessary, allowing space for null terminator
      if (n + chunk + 1 > cap) {
        cap = max(n + chunk + 1, cap * 2);
        s = xrealloc(s, cap);
      }

      // read another chunk
      si r = read(0, s + n, chunk);
      if (r < 0) {
        perror("read");
        exit(1);
      }
      n += r;

      // no more data to read
      if (r != chunk)
        break;
    }
  }

  // null terminator
  s[n] = 0;

  // start at the beginning
  txt = txtstart = s;
}

noret ferr(const char *msg) {
  // line number
  si line = 1;
  for (char *s = txtstart; s != tokstart; ++s)
    if (*s == '\n')
      line++;

  // start of line
  char *linestart = tokstart;
  while (!(linestart == txtstart || linestart[-1] == '\n'))
    linestart--;

  // print context
  for (char *s = linestart; *s >= ' ' || *s == '\t'; ++s)
    fputc(*s, stderr);
  fputc('\n', stderr);

  // print caret
  for (char *s = linestart; s != tokstart; ++s)
    fputc(*s == '\t' ? '\t' : ' ', stderr);
  fprintf(stderr, "^\n");

  // print message and exit
  fprintf(stderr, "%s:%zu: %s\n", file, line, msg);
  exit(1);
}

// tokenizer
enum {
  k_term = 1,
};

char isid[0x100];

void init_parser(void) {
  // SORT
  isid['*'] = 1;
  isid['+'] = 1;
  isid['-'] = 1;
  isid['/'] = 1;
  isid['<'] = 1;
  isid['='] = 1;
  isid['>'] = 1;
  isid['?'] = 1;
  isid['_'] = 1;
  memset(isid + '0', 1, 10);
  memset(isid + 'A', 1, 26);
  memset(isid + 'a', 1, 26);
  ///
}

static int xescape(char **sp, int n) {
  char *s = *sp;
  int c = 0;
  for (int i = 0; i < n; i++) {
    int d = xdigit(*s++);
    if (d < 0)
      break;
    c = c * 16 + d;
  }
  *sp = s;
  return c;
}

static void quote(vec *v) {
  char *s = txt;
  int q = *s++;
  vinit(v);
  while (*s != q) {
    si c = *s++;
    switch (c) {
    case '\\':
      c = *s++;
      switch (c) {
      case '"':
      case '?':
      case '\'':
      case '\\':
        break;
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
        s--;
        c = 0;
        for (int i = 0; i < 3; i++) {
          if (!('0' <= *s && *s <= '7'))
            break;
          c = c * 8 + *s++ - '0';
        }
        break;
      case 'U':
        c = xescape(&s, 8);
        break;
      case 'a':
        c = '\a';
        break;
      case 'b':
        c = '\b';
        break;
      case 'f':
        c = '\f';
        break;
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      case 'u':
        c = xescape(&s, 4);
        break;
      case 'v':
        c = '\v';
        break;
      case 'x':
        c = xescape(&s, 2);
        break;
      default:
        ferr("unknown escape character");
      }
      break;
    case '\n':
    case '\r':
    case 0:
      ferr("unclosed quote");
    }
    vpush(v, mkint(c));
  }
  txt = s + 1;
  tok = k_term;
}

static void lexexact(char *s, int base) {
  while (isid[*s])
    s++;
  // mpq_set_str does not tolerate non-whitespace after the number, so it must
  // be null terminated
  int c = *s;
  *s = 0;
  Rat r;
  mpq_init(r.val);
  int e = mpq_set_str(r.val, txt, 0);
  // and restore the byte we overwrote
  *s = c;
  if (e)
    ferr("invalid number");
  txt = s;
  tokterm = irat(&r);
}

static void lexinexact(void) {
  errno = 0;
  char *end = 0;
  // strtod does tolerate non-whitespace after the number, and will tell us
  // where the number ended
  double r = strtod(txt, &end);
  if (errno || !end)
    ferr(strerror(errno));
  txt = end;
  tokterm = ifloat(r);
  return;
}

static void num(void) {
  char *s = txt;
  tok = k_term;

  // sign
  if (*s == '-')
    s++;

  // explicit base must be handled first so the rest of the code can assume E is
  // not a valid digit and therefore indicates a floating-point number
  if (*s == '0')
    switch (s[1]) {
    case 'X':
    case 'x':
      s += 2;
      while (isxdigit1(*s))
        s++;
      switch (*s) {
      case '.':
      case 'P':
      case 'p':
        lexinexact();
        return;
      }
      // fallthru
    case 'B':
    case 'b':
      lexexact(s, 0);
      return;
    }

  // now the integer part contains only decimal digits
  while (isdigit1(*s))
    s++;

  // floating-point number
  switch (*s) {
  case '.':
  case 'E':
  case 'e':
    lexinexact();
    return;
  }

  // mpq_set_str would take leading 0 for octal, but that's not desirable in a
  // language that does not attempt to be compatible with C, so specify the base
  // as 10
  lexexact(s, 10);
}

static void lex(void) {
loop:
  char *s = tokstart = txt;
  vec v;
  switch (*s) {
  case ' ':
  case '\f':
  case '\n':
  case '\r':
  case '\t':
  case '\v':
    txt = s + 1;
    goto loop;
  case '"':
    quote(&v);
    tokterm = list2(mkeyword(w_quote), list(&v));
    return;
  case '-':
    if (isdigit1(s[1])) {
      num();
      return;
    }
    // fallthru
  case '*':
  case '+':
  case '/':
  case '<':
  case '=':
  case '>':
  case '?':
  case 'A':
  case 'B':
  case 'C':
  case 'D':
  case 'E':
  case 'F':
  case 'G':
  case 'H':
  case 'I':
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
  case 'O':
  case 'P':
  case 'Q':
  case 'R':
  case 'S':
  case 'T':
  case 'U':
  case 'V':
  case 'W':
  case 'X':
  case 'Y':
  case 'Z':
  case '_':
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  case 'e':
  case 'f':
  case 'g':
  case 'h':
  case 'i':
  case 'j':
  case 'k':
  case 'l':
  case 'm':
  case 'n':
  case 'o':
  case 'p':
  case 'q':
  case 'r':
  case 's':
  case 't':
  case 'u':
  case 'v':
  case 'w':
  case 'x':
  case 'y':
  case 'z':
    assert(isid[*s]);
    do
      s++;
    while (isid[*s]);
    tok = k_term;
    tokterm = intern(txt, s - txt);
    txt = s;
    return;
  case '.':
    if (isdigit1(s[1])) {
      num();
      return;
    }
    break;
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
    num();
    return;
  case ';':
    txt = strchr(s, '\n');
    if (!txt)
      txt = "";
    goto loop;
  case '\'':
    quote(&v);
    if (v.n != 1)
      ferr("expected one character");
    tokterm = *v.p;
    return;
  case '{':
    txt = strchr(s, '}');
    if (!txt)
      ferr("unclosed '{'");
    txt++;
    goto loop;
  case 0:
    tok = 0;
    return;
  }
  assert(!isid[*s]);
  txt = s + 1;
  tok = *s;
}

static int eat(int k) {
  if (tok != k)
    return 0;
  lex();
  return 1;
}

// parser
static si expr(void) {
  vec v;
  switch (tok) {
  case '#':
    lex();
    return list2(mkeyword(w_quote), expr());
  case '(':
    lex();
    vinit(&v);
    while (!eat(')'))
      vpush(&v, expr());
    return list(&v);
  case ',':
    lex();
    return list2(mkeyword(w_unquote), expr());
  case '@':
    lex();
    return list2(mkeyword(w_unquotes), expr());
  case '[':
    lex();
    vinit(&v);
    while (!eat(']'))
      vpush(&v, expr());
    return list(&v);
  case '\\':
    lex();
    return mkeyword(w_lambda);
  case k_term: {
    si a = tokterm;
    lex();
    return a;
  }
  }
  ferr("expected expression");
}

void parse(vec *v) {
  lex();
  while (tok)
    vpush(v, expr());
}
