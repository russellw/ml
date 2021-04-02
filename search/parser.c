#include "main.h"
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#define O_BINARY 0
#endif

char *file;
char *txtstart;
char *txt;
char *tokstart;
int bufi;
int tok;

void readfile(char *file0) {
  file = file0;
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

enum {
  k_sym,
};

// tokenizer
void pushc(char c) {
  if (bufi == sizeof buf - 1)
    err("token too long");
  buf[bufi++] = c;
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
  case ';':
    txt = strchr(s, '\n');
    goto loop;
  case 0:
    tok = 0;
    return;
  }
  txt = s + 1;
  tok = *s;
}
