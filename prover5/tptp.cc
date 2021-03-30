#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

// as far as can be measured, checking for word versus separator characters with
// a lookup table that fits in primary cache, is several times faster than doing
// it with multiple comparisons and branches
char isWord[0x100];

namespace {
enum {
  t_distinctObj = 1,
  t_dollarWord,
  t_eqv,
  t_imp,
  t_impr,
  t_int,
  t_nand,
  t_ne,
  t_nor,
  t_rat,
  t_real,
  t_var,
  t_word,
  t_xor,
};

#ifdef DEBUG
si header;
#endif

struct init {
  init() {
    memset(isWord + '0', 1, 10);
    memset(isWord + 'A', 1, 26);
    isWord['_'] = 1;
    memset(isWord + 'a', 1, 26);
  }
} init1;

struct selection : unordered_set<sym *> {
  bool all;

  explicit selection(bool all) : all(all) {}

  si count(sym *s) const {
    if (all)
      return 1;
    return unordered_set<sym *>::count(s);
  }
};

void strmemcpy(char *dest, const char *src, const char *e) {
  auto n = e - src;
  memcpy(dest, src, n);
  dest[n] = 0;
}

struct parser1 : parser {
  // SORT
  bool cnfMode;
  selection sel;
  vec<pair<sym *, term>> vars;
  ///

  // tokenizer
  void word() {
    auto s = text;
    while (isWord[*s])
      ++s;
    tokSym = intern(text, s - text);
    text = s;
  }

  void quote() {
    auto s = text;
    auto q = *s++;
    si i = 0;
    while (*s != q) {
      if (*s == '\\')
        ++s;
      if (*s < ' ')
        err("unclosed quote");
      if (i >= sizeof buf)
        err("symbol too long");
      buf[i++] = *s++;
    }
    text = s + 1;
    tokSym = intern(buf, i);
  }

  void sign() {
    switch (*text) {
    case '+':
    case '-':
      ++text;
      break;
    }
  }

  void digits() {
    auto s = text;
    while (isDigit(*s))
      ++s;
    text = s;
  }

  void exp() {
    assert(*text == 'E' || *text == 'e');
    ++text;
    sign();
    digits();
  }

  void num() {
    sign();
    // gmp doesn't handle unary +, so need to omit it from token
    if (*tokStart == '+')
      ++tokStart;
    // sign without digits should give a clear error message
    if (!isDigit(*text))
      err("expected digit", text);
    tok = t_int;
    digits();
    switch (*text) {
    case '.':
      tok = t_real;
      ++text;
      digits();
      switch (*text) {
      case 'E':
      case 'e':
        exp();
        break;
      }
      break;
    case '/':
      tok = t_rat;
      ++text;
      digits();
      break;
    case 'E':
    case 'e':
      tok = t_real;
      exp();
      break;
    }
    if (text - tokStart > sizeof buf - 1)
      err("number too long");
  }

  void lex() {
  loop:
    auto s = tokStart = text;
    switch (*s) {
    case ' ':
    case '\f':
    case '\n':
    case '\r':
    case '\t':
    case '\v':
      text = s + 1;
      goto loop;
    case '!':
      switch (s[1]) {
      case '=':
        text = s + 2;
        tok = t_ne;
        return;
      }
      break;
    case '"':
      tok = t_distinctObj;
      quote();
      return;
    case '$':
      text = s + 1;
      tok = t_dollarWord;
      word();
      return;
    case '%': {
      text = strchr(s, '\n');
#ifdef DEBUG
      if (expected == szs::none) {
        string s1(s, text);
        smatch m;
        if (regex_match(s1, m, regex(R"(% Status\s*:\s*(\term+)\s*)"))) {
          for (si i = 1; i != (si)szs::max; ++i)
            if (m[1] == szsNames[i]) {
              expected = (szs)i;
              break;
            }
          if (expected == szs::none)
            err("unknown status");
        }
      }
      if (header) {
        if (s[1] == '-' && s[2] == '-')
          --header;
        while (s != text)
          putchar(*s++);
        putchar('\n');
        if (s[1] == '\n')
          putchar('\n');
      }
#endif
      goto loop;
    }
    case '+':
    case '-':
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
    case '/':
      if (s[1] != '*') {
        text = s + 1;
        err("expected '*'");
      }
      for (s += 2; !(s[0] == '*' && s[1] == '/'); ++s)
        if (!*s)
          err("unclosed comment");
      text = s + 2;
      goto loop;
    case '<':
      switch (s[1]) {
      case '=':
        if (s[2] == '>') {
          text = s + 3;
          tok = t_eqv;
          return;
        }
        text = s + 2;
        tok = t_impr;
        return;
      case '~':
        if (s[2] == '>') {
          text = s + 3;
          tok = t_xor;
          return;
        }
        tokStart = s + 2;
        err("expected '>'");
      }
      break;
    case '=':
      switch (s[1]) {
      case '>':
        text = s + 2;
        tok = t_imp;
        return;
      }
      break;
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
      tok = t_var;
      word();
      return;
    case '\'':
      tok = t_word;
      quote();
      return;
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
      tok = t_word;
      word();
      return;
    case '~':
      switch (s[1]) {
      case '&':
        text = s + 2;
        tok = t_nand;
        return;
      case '|':
        text = s + 2;
        tok = t_nor;
        return;
      }
      break;
    case 0:
      tok = 0;
      return;
    }
    text = s + 1;
    tok = *s;
  }

  bool eat(char o) {
    if (tok == o) {
      lex();
      return 1;
    }
    return 0;
  }

  void expect(char o) {
    if (eat(o))
      return;
    sprintf(buf, "expected '%c'", o);
    err(buf);
  }

  void expect(char o, const char *s) {
    if (eat(o))
      return;
    sprintf(buf, "expected %s", s);
    err(buf);
  }

  // types
  type atomicType() {
    auto k = tok;
    auto s = tokSym;
    auto ts = tokStart;
    lex();
    switch (k) {
    case '!':
    case '[':
      throw inappropriate();
    case '(': {
      auto t = atomicType();
      expect(')');
      return t;
    }
    case t_dollarWord:
      switch (keyword(s)) {
      case k_i:
        return type::Individual;
      case k_int:
        return type::Int;
      case k_o:
        return type::Bool;
      case k_rat:
        return type::Rat;
      case k_real:
        return type::Real;
      }
      throw inappropriate();
    case t_word:
      return internType(s);
    default:
      err("expected type", ts);
    }
  }

  type topLevelType() {
    if (eat('(')) {
      vec<type> v(1);
      do
        v.push_back(atomicType());
      while (eat('*'));
      expect(')');
      expect('>');
      v[0] = atomicType();
      return internType(v);
    }
    auto t = atomicType();
    if (eat('>'))
      return internType(atomicType(), t);
    return t;
  }

  // terms
  term parseInt() {
    strmemcpy(buf, tokStart, text);
    Int x;
    if (mpz_init_set_str(x.val, buf, 10))
      err("invalid number");
    lex();
    return tag(term::Int, intern(x));
  }

  term parseRat() {
    strmemcpy(buf, tokStart, text);
    Rat x;
    mpq_init(x.val);
    if (mpq_set_str(x.val, buf, 10))
      err("invalid number");
    lex();
    return tag(term::Rat, intern(x));
  }

  term parseReal() {
    auto p = tokStart;

    // sign
    bool sign = 0;
    if (*p == '-') {
      ++p;
      sign = 1;
    }

    // integer part
    auto q = p;
    while (isDigit(*q))
      ++q;
    strmemcpy(buf, p, q);
    mpz_t integer;
    mpz_init_set_str(integer, buf, 10);
    p = q;

    // decimal part
    mpz_t mantissa;
    mpz_init(mantissa);
    si scale = 0;
    if (*p == '.') {
      ++p;
      q = p;
      while (isDigit(*q))
        ++q;
      strmemcpy(buf, p, q);
      mpz_set_str(mantissa, buf, 10);
      scale = q - p;
      p = q;
    }
    mpz_t powScale;
    mpz_init(powScale);
    mpz_ui_pow_ui(powScale, 10, scale);

    // mantissa += integer * 10^scale
    mpz_addmul(mantissa, integer, powScale);

    // sign
    if (sign)
      mpz_neg(mantissa, mantissa);

    // result = scaled mantissa
    Rat x;
    mpq_init(x.val);
    mpq_set_num(x.val, mantissa);
    mpq_set_den(x.val, powScale);

    // exponent
    bool exponentSign = 0;
    si exponent = 0;
    if (*p == 'e' || *p == 'e') {
      ++p;
      switch (*p) {
      case '-':
        exponentSign = 1;
      case '+':
        ++p;
        break;
      }
      errno = 0;
      exponent = strtoul(p, 0, 10);
      if (errno)
        err(strerror(errno));
    }
    mpz_t powExponent;
    mpz_init(powExponent);
    mpz_ui_pow_ui(powExponent, 10, exponent);
    if (exponentSign)
      mpz_mul(mpq_denref(x.val), mpq_denref(x.val), powExponent);
    else
      mpz_mul(mpq_numref(x.val), mpq_numref(x.val), powExponent);

    // cleanup
    mpz_clear(powExponent);
    mpz_clear(powScale);
    mpz_clear(mantissa);
    mpz_clear(integer);

    // result
    lex();
    return tag(term::Real, intern(x));
  }

  void args(vec<term> &v) {
    expect('(');
    do
      v.push_back(atomicTerm());
    while (eat(','));
    expect(')');
  }

  void args(vec<term> &v, si arity) {
    auto old = v.n;
    args(v);
    if (v.n - old == arity)
      return;
    sprintf(buf, "expected %zu arguments", arity);
    err(buf);
  }

  term definedFunctor(term op, si arity) {
    vec<term> v;
    args(v, arity);
    auto t = typeofNum(v[0]);
    for (auto i = v.p + 1, e = v.end(); i != e; ++i)
      requireType(t, *i);
    return intern(op, v);
  }

  term atomicTerm() {
    switch (tok) {
    case '!':
      throw inappropriate();
    case t_distinctObj: {
      auto a = tag(term::DistinctObj, tokSym);
      lex();
      return a;
    }
    case t_dollarWord: {
      auto s = tokSym;
      auto ts = tokStart;
      lex();
      vec<term> v;
      switch (keyword(s)) {
      case k_ceiling:
        return definedFunctor(term::Ceil, 1);
      case k_difference:
        return definedFunctor(term::Sub, 2);
      case k_distinct: {
        args(v);
        defaultType(type::Individual, v[0]);
        auto t = typeof(v[0]);
        for (auto i = v.p + 1, e = v.end(); i != e; ++i)
          requireType(t, *i);
        vec<term> inequalities;
        for (auto i = v.p, e = v.end(); i != e; ++i)
          for (auto j = v.p; j != i; ++j)
            inequalities.push_back(intern(term::Not, intern(term::Eq, *i, *j)));
        return intern(term::And, inequalities);
      }
      case k_false:
        return term::False;
      case k_floor:
        return definedFunctor(term::Floor, 1);
      case k_greater: {
        args(v, 2);
        auto t = typeofNum(v[0]);
        requireType(t, v[1]);
        return intern(term::Lt, v[1], v[0]);
      }
      case k_greatereq: {
        args(v, 2);
        auto t = typeofNum(v[0]);
        requireType(t, v[1]);
        return intern(term::Le, v[1], v[0]);
      }
      case k_is_int:
        return definedFunctor(term::IsInt, 1);
      case k_is_rat:
        return definedFunctor(term::IsRat, 1);
      case k_ite:
        throw inappropriate();
      case k_less:
        return definedFunctor(term::Lt, 2);
      case k_lesseq:
        return definedFunctor(term::Le, 2);
      case k_product:
        return definedFunctor(term::Mul, 2);
      case k_quotient: {
        auto a = definedFunctor(term::Div, 2);
        if (typeof(at(a, 1)) == type::Int)
          err("expected fraction term");
        return a;
      }
      case k_quotient_e:
        return definedFunctor(term::DivE, 2);
      case k_quotient_f:
        return definedFunctor(term::DivF, 2);
      case k_quotient_t:
        return definedFunctor(term::DivT, 2);
      case k_remainder_e:
        return definedFunctor(term::RemE, 2);
      case k_remainder_f:
        return definedFunctor(term::RemF, 2);
      case k_remainder_t:
        return definedFunctor(term::RemT, 2);
      case k_round:
        return definedFunctor(term::Round, 1);
      case k_sum:
        return definedFunctor(term::Add, 2);
      case k_to_int:
        return definedFunctor(term::ToInt, 1);
      case k_to_rat:
        return definedFunctor(term::ToRat, 1);
      case k_to_real:
        return definedFunctor(term::ToReal, 1);
      case k_true:
        return term::True;
      case k_truncate:
        return definedFunctor(term::Trunc, 1);
      case k_uminus:
        return definedFunctor(term::Minus, 1);
      }
      err("unknown word", ts);
    }
    case t_int:
      return parseInt();
    case t_rat:
      return parseRat();
    case t_real:
      return parseReal();
    case t_var: {
      auto s = tokSym;
      auto ts = tokStart;
      lex();
      for (auto i = vars.rbegin(), e = vars.rend(); i != e; ++i)
        if (i->first == s)
          return i->second;
      if (!cnfMode)
        err("unknown variable", ts);
      auto x = var(type::Individual, vars.n);
      vars.push_back(make_pair(s, x));
      return x;
    }
    case t_word: {
      auto a = tag(term::Sym, tokSym);
      lex();
      if (tok != '(')
        return a;
      vec<term> v(1);
      v[0] = a;
      args(v);
      for (auto i = v.p + 1, e = v.end(); i != e; ++i)
        defaultType(type::Individual, *i);
      return intern(term::Call, v);
    }
    }
    err("syntax error");
  }

  term infixUnary() {
    auto a = atomicTerm();
    switch (tok) {
    case '=': {
      lex();
      auto b = atomicTerm();
      defaultType(type::Individual, a);
      requireType(typeof(a), b);
      return intern(term::Eq, a, b);
    }
    case t_ne: {
      lex();
      auto b = atomicTerm();
      defaultType(type::Individual, a);
      requireType(typeof(a), b);
      return intern(term::Not, intern(term::Eq, a, b));
    }
    }
    requireType(type::Bool, a);
    return a;
  }

  term quantifiedFormula(term op) {
    lex();
    expect('[');
    auto old = vars.n;
    vec<term> v(1);
    do {
      if (tok != t_var)
        err("expected variable");
      auto s = tokSym;
      lex();
      auto t = type::Individual;
      if (eat(':'))
        t = atomicType();
      auto x = var(t, vars.n);
      vars.push_back(make_pair(s, x));
      v.push_back(x);
    } while (eat(','));
    expect(']');
    expect(':');
    v[0] = unitaryFormula();
    vars.n = old;
    return intern(op, v);
  }

  term unitaryFormula() {
    switch (tok) {
    case '!':
      return quantifiedFormula(term::All);
    case '(': {
      lex();
      auto a = logicFormula();
      expect(')');
      return a;
    }
    case '?':
      return quantifiedFormula(term::Exists);
    case '~':
      lex();
      return intern(term::Not, unitaryFormula());
    }
    return infixUnary();
  }

  term associativeLogicFormula(term op, term a) {
    vec<term> v(1);
    v[0] = a;
    auto o = tok;
    while (eat(o))
      v.push_back(unitaryFormula());
    return intern(op, v);
  }

  term logicFormula() {
    auto a = unitaryFormula();
    switch (tok) {
    case '&':
      return associativeLogicFormula(term::And, a);
    case '|':
      return associativeLogicFormula(term::Or, a);
    case t_eqv:
      lex();
      return intern(term::Eqv, a, unitaryFormula());
    case t_imp:
      lex();
      return intern(term::Imp, a, unitaryFormula());
    case t_impr:
      lex();
      return intern(term::Imp, unitaryFormula(), a);
    case t_nand:
      lex();
      return intern(term::Not, intern(term::And, a, unitaryFormula()));
    case t_nor:
      lex();
      return intern(term::Not, intern(term::Or, a, unitaryFormula()));
    case t_xor:
      lex();
      return intern(term::Not, intern(term::Eqv, a, unitaryFormula()));
    }
    return a;
  }

  // top level
  sym *name() {
    switch (tok) {
    case t_int: {
      auto s = intern(tokStart, text - tokStart);
      lex();
      return s;
    }
    case t_word: {
      auto s = tokSym;
      lex();
      return s;
    }
    }
    err("expected name");
  }

  void ignore() {
    switch (tok) {
    case '(':
      lex();
      while (!eat(')'))
        ignore();
      return;
    case 0:
      err("unexpected end of file");
    }
    lex();
  }

  parser1(const char *file, const selection &sel) : parser(file), sel(sel) {
    try {
      lex();
      while (tok) {
        auto ts = tokStart;
        vars.n = 0;
        switch (keyword(name())) {
        case k_cnf: {
          expect('(');

          // name
          auto clauseName = name();
          expect(',');

          // role
          name();
          expect(',');

          // literals
          cnfMode = 1;
          vec<term> neg, pos;
          auto parens = eat('(');
          do {
            auto no = eat('~');
            auto a = infixUnary();
            ck(a);
            if (tag(a) == term::Not) {
              no = no ^ 1;
              a = at(a, 0);
            }
            (no ? neg : pos).push_back(a);
          } while (eat('|'));
          if (parens)
            expect(')');

          // select
          if (!sel.count(clauseName))
            break;

          // clause
          auto c = input(neg, pos, how::none);
          setFile(c, file);
          setName(c, clauseName->v);
          break;
        }
        case k_fof:
        case k_tff: {
          expect('(');

          // name
          auto formulaName = name();
          expect(',');

          // role
          if (tok != t_word)
            err("expected role");
          auto role = keyword(tokSym);
          if (role == k_conjecture && conjecture)
            err("multiple conjectures not supported");
          lex();
          expect(',');

          // type
          if (role == k_type) {
            si parens = 0;
            while (eat('('))
              ++parens;
            auto s = name();
            expect(':');
            ts = tokStart;
            if (tok == t_dollarWord && tokSym == keywords + k_tType) {
              lex();
              if (tok == '>')
                throw inappropriate();
            } else {
              auto t = topLevelType();
              if (s->t == type::none)
                s->t = t;
              else if (t != s->t)
                err("type mismatch");
            }
            while (parens--)
              expect(')');
            break;
          }

          // formula
          cnfMode = 0;
          auto a = logicFormula();
          assert(!vars.n);
          ck(a);

          // select
          if (!sel.count(formulaName))
            break;

          auto f = mk(a, how::none);
          setFile(f, file);
          setName(f, formulaName->v);
          if (role == k_conjecture) {
            a = intern(term::Not, a);
            f = mk(a, how::negate, f);
            conjecture = f;
          }

          // cnf
          cnf(a, f);
          break;
        }
        case k_include: {
          auto dir = getenv("TPTP");
          if (!dir)
            err("TPTP environment variable not set", ts);
          expect('(');

          // file
          snprintf(buf, sizeof buf, "%s/%s", dir, name()->v);
          auto file1 = intern(buf, strlen(buf))->v;

          // select and read
          if (eat(',')) {
            expect('[');
            selection sel1(0);
            do {
              auto selName = name();
              if (sel.count(selName))
                sel1.insert(selName);
            } while (eat(','));
            expect(']');
            parser1 p(file1, sel1);
          } else {
            parser1 p(file1, sel);
          }
          break;
        }
        default:
          err("unknown language", ts);
        }
        if (tok == ',')
          do
            ignore();
          while (tok != ')');
        expect(')');
        expect('.');
      }
    } catch (const char *e) {
      err(e);
    }
  }
};
} // namespace

void tptp(const char *file) {
#ifdef DEBUG
  header = 2;
#endif
  parser1 p(file, selection(1));
}
