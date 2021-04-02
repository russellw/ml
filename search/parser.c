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

    // allow space for extra newline if required, and null terminator
    s = (char *)xmalloc(n + 2);

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
      // expand buffer as necessary, allowing space for extra newline if
      // required, and null terminator
      if (n + chunk + 2 > cap) {
        cap = max(n + chunk + 2, cap * 2);
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

  // newline and null terminator
  s[n] = 0;
  if (n && s[n - 1] != '\n') {
    s[n] = '\n';
    s[n + 1] = 0;
  }

  // start at the beginning
  txtstart = s;
  txt = s;
  tokstart = s;
}

// tokenizer
enum {
  k_char = 1,
  k_id,
  k_str,
  k_sym,
  k_term,
};

char isid[0x100];

void init_parser(void) {
  // SORT
  memset(isid + '0', 1, 10);
  memset(isid + 'A', 1, 26);
  memset(isid + 'a', 1, 26);
  isid['_'] = 1;
  isid['+'] = 1;
  isid['-'] = 1;
  isid['*'] = 1;
  isid['/'] = 1;
  isid['<'] = 1;
  isid['>'] = 1;
  isid['='] = 1;
  isid['?'] = 1;
  ///
}

static int xdigit(int c) {
  if (isdigit1(c))
    return c - '0';
  if ('a' <= c && c <= 'f')
    return c - 'a' + 10;
  if ('A' <= c && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

void pushc(int c) {
  if (bufi == sizeof buf - 1)
    err("token too long");
  buf[bufi++] = c;
}

static void quote(void) {
  char *s = txt;
  int q = *s++;
  bufi = 0;
  while (*s != q) {
    int c = *s++;
    switch (c) {
    case '\n':
      err("unclosed quote");
    case '\\':
      c = *s++;
      switch (c) {
      case '\\':
      case '\'':
      case '"':
      case '?':
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
      case 'v':
        c = '\v';
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
      case 'x':
        c = 0;
        for (int i = 0; i < 2; i++) {
          int d = hexdigit(*s++);
          if (d < 0)
            break;
          c = c * 16 + d;
        }
        break;
      default:
        err("unknown escape character");
      }
      break;
    }
    pushc(c);
  }
  txt = s + 1;
}

static void num(void) {
  char *s = txt;
  bufi = 0;
  if (*s == '-')
    pushc(*s++);
  Int int1;
  if (*s == '0')
    switch (s[1]) {
    case 'b':
    case 'B':
      s += 2;
      while (isid[*s])
        pushc(*s++);
      if (mpz_init_set_str(int1.val, buf, 2))
        err("invalid binary integer");
      tok = k_term;
      tokterm = iint(&int1);
    }
  while (isdigit1(*s))
    s++;
  switch (*s) {
  case '/':
    do
      s++;
    while (isdigit1(*s));
  }
}

static void lex(void) {
loop:
  char *s = tokstart = txt;
  switch (*s) {
  case ' ':
  case '\f':
  case '\n':
  case '\r':
  case '\t':
  case '\v':
    txt = s + 1;
    goto loop;
  case '\'':
    tok = k_char;
    quote();
    return;
  case '"':
    tok = k_str;
    quote();
    return;
  case ';':
    txt = strchr(s, '\n');
    goto loop;
  case 0:
    tok = 0;
    return;
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
  case '_':
  case '+':
  case '-':
  case '*':
  case '/':
  case '<':
  case '>':
  case '=':
  case '?':
    assert(isid[*s]);
    do
      s++;
    while (isid[*s]);
    txt = s;
    tok = k_id;
    return;
  }
  assert(!isid[*s]);
  txt = s + 1;
  tok = *s;
}
