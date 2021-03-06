#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

namespace {
enum {
  t_num = 1,
  t_zero,
};

struct parser1 : parser {
  // tokenizer
  void lex() {
  loop:
    auto s = tokStart = text;
    switch (*text) {
    case ' ':
    case '\f':
    case '\n':
    case '\r':
    case '\t':
    case '\v':
      text = s + 1;
      goto loop;
    case '0':
      if (!isDigit(s[1])) {
        text = s + 1;
        tok = t_zero;
        return;
      }
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      do
        ++s;
      while (isDigit(*s));
      text = s;
      tokSym = intern(tokStart, s - tokStart);
      tok = t_num;
      return;
    case 'c': {
      text = strchr(s, '\n');
#ifdef DEBUG
      string line(s, text);
      smatch m;
      if (expected == szs::none &&
          regex_match(line, m, regex(R"(c .* (SAT|UNSAT) .*)")))
        expected = m[1] == "SAT" ? szs::Satisfiable : szs::Unsatisfiable;
#endif
      goto loop;
    }
    case 0:
      tok = 0;
      return;
    }
    text = s + 1;
    tok = *s;
  }

  // terms
  term fn() {
    auto s = tokSym;
    lex();
    s->t = type::Bool;
    return tag(term::Sym, s);
  }

  // top level
  parser1(const char *file) : parser(file) {
    try {
      lex();
      if (tok == 'p') {
        while (isSpace(*text))
          ++text;

        if (!(text[0] == 'c' && text[1] == 'n' && text[2] == 'f'))
          err("expected 'cnf'");
        text += 3;
        lex();

        if (tok != t_num)
          err("expected count");
        lex();

        if (tok != t_num)
          err("expected count");
        lex();
      }
      vec<term> neg, pos;
      for (;;)
        switch (tok) {
        case '-':
          neg.push_back(fn());
          break;
        case 0:
          if (neg.n | pos.n) {
            auto c = input(neg, pos, how::none);
            setFile(c, file);
          }
          return;
        case t_num:
          pos.push_back(fn());
          break;
        case t_zero: {
          lex();
          auto c = input(neg, pos, how::none);
          neg.n = pos.n = 0;
          setFile(c, file);
          break;
        }
        default:
          err("syntax error");
        }
    } catch (const char *e) {
      err(e);
    }
  }
};
} // namespace

void dimacs(const char *file) { parser1 p(file); }
