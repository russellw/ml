package prover;

import io.vavr.collection.Array;
import io.vavr.collection.HashMap;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import java.io.*;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.regex.Pattern;

public final class TptpParser {
  private static final Pattern STATUS_PATTERN = Pattern.compile("\\s*Status\\s*:\\s*(\\w+)");

  // Tokens
  private static final int DEFINED_WORD = -2;
  private static final int DISTINCT_OBJECT = -3;
  private static final int EQV = -4;
  private static final int IMPLIES = -5;
  private static final int IMPLIESR = -6;
  private static final int INTEGER = -7;
  private static final int NAND = -8;
  private static final int NOT_EQ = -9;
  private static final int NOR = -10;
  private static final int RATIONAL = -11;
  private static final int REAL = -12;
  private static final int WORD = -14;
  private static final int XOR = -15;
  private static final int VARIABLE = -16;

  // Problem state
  private static java.util.HashMap<String, Func> types = new java.util.HashMap<>();
  private static java.util.HashMap<String, Func> funcs;
  private static Problem problem;

  // File state
  private final String file;
  private final LineNumberReader reader;
  private int c;
  private boolean header;
  private int token;
  private String tokenString;
  private java.util.HashMap<String, Variable> free = new java.util.HashMap<>();

  // Tokenizer
  private void lexQuote() throws IOException {
    var line = reader.getLineNumber();
    var quote = c;
    c = reader.read();
    var sb = new StringBuilder();
    while (c != quote) {
      if (c < ' ') throw new ParseException(file, line, "unclosed quote");
      if (c == '\\') c = reader.read();
      sb.append((char) c);
      c = reader.read();
    }
    c = reader.read();
    tokenString = sb.toString();
  }

  private String lexWord() throws IOException {
    var sb = new StringBuilder();
    do {
      sb.append((char) c);
      c = reader.read();
    } while (Character.isJavaIdentifierPart(c));
    return sb.toString();
  }

  private void lex() throws IOException {
    for (; ; ) {
      token = c;
      switch (c) {
        case ' ':
        case '\f':
        case '\n':
        case '\r':
        case '\t':
          c = reader.read();
          continue;
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
          {
            var sb = new StringBuilder();
            do {
              sb.append((char) c);
              c = reader.read();
            } while (Character.isDigit(c));
            switch (c) {
              case '.':
                do {
                  sb.append((char) c);
                  c = reader.read();
                } while (Character.isDigit(c));
                break;
              case '/':
                do {
                  sb.append((char) c);
                  c = reader.read();
                } while (Character.isDigit(c));
                token = RATIONAL;
                tokenString = sb.toString();
                return;
              case 'E':
              case 'e':
                break;
              default:
                token = INTEGER;
                tokenString = sb.toString();
                return;
            }
            if (c == 'e' || c == 'E') {
              sb.append((char) c);
              c = reader.read();
            }
            if (c == '+' || c == '-') {
              sb.append((char) c);
              c = reader.read();
            }
            while (Character.isDigit(c)) {
              sb.append((char) c);
              c = reader.read();
            }
            token = REAL;
            tokenString = sb.toString();
            return;
          }
        case '/':
          var line = reader.getLineNumber();
          c = reader.read();
          if (c != '*') {
            throw new ParseException(file, reader.getLineNumber(), "'*' expected");
          }
          do {
            do {
              if (c == -1) {
                throw new ParseException(file, line, "unclosed block comment");
              }
              c = reader.read();
            } while (c != '*');
            c = reader.read();
          } while (c != '/');
          c = reader.read();
          continue;
        case '!':
          c = reader.read();
          if (c == '=') {
            c = reader.read();
            token = NOT_EQ;
            return;
          }
          return;
        case '<':
          c = reader.read();
          if (c == '=') {
            c = reader.read();
            if (c == '>') {
              c = reader.read();
              token = EQV;
              return;
            }
            token = IMPLIESR;
            return;
          }
          if (c == '~') {
            c = reader.read();
            if (c == '>') {
              c = reader.read();
              token = XOR;
              return;
            }
            throw new ParseException(file, reader.getLineNumber(), "'>' expected");
          }
          return;
        case '=':
          c = reader.read();
          if (c == '>') {
            c = reader.read();
            token = IMPLIES;
            return;
          }
          return;
        case '~':
          c = reader.read();
          if (c == '&') {
            c = reader.read();
            token = NAND;
            return;
          }
          if (c == '|') {
            c = reader.read();
            token = NOR;
            return;
          }
          return;
        case '%':
          {
            var s = reader.readLine();
            c = reader.read();
            if (header) {
              problem.header.add('%' + s);
              if (c == '\n') problem.header.add("");
            }
            if (problem.expected == null) {
              var matcher = STATUS_PATTERN.matcher(s);
              if (matcher.matches()) problem.expected = SZS.valueOf(matcher.group(1));
            }
            continue;
          }
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
          token = VARIABLE;
          tokenString = lexWord();
          return;
        case '\'':
          token = WORD;
          lexQuote();
          return;
        case '"':
          token = DISTINCT_OBJECT;
          lexQuote();
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
          token = WORD;
          tokenString = lexWord();
          return;
        case '$':
          token = DEFINED_WORD;
          tokenString = lexWord();
          return;
      }
      c = reader.read();
      return;
    }
  }

  private boolean eat(int k) throws IOException {
    if (token == k) {
      lex();
      return true;
    }
    return false;
  }

  private void expect(int k) throws IOException {
    if (!eat(k))
      throw new ParseException(file, reader.getLineNumber(), ": '" + (char) k + "' expected");
  }

  // Types
  private Object atomicType() throws IOException {
    var k = token;
    var s = tokenString;
    lex();
    switch (k) {
      case '(':
        {
          var type = atomicType();
          expect(')');
          return type;
        }
      case '!':
      case '[':
        throw new InappropriateException();
      case DEFINED_WORD:
        switch (s) {
          case "$i":
            return Symbol.INDIVIDUAL;
          case "$int":
            return Symbol.INTEGER;
          case "$o":
            return Symbol.BOOLEAN;
          case "$rat":
            return Symbol.RATIONAL;
          case "$real":
            return Symbol.REAL;
          case "$tType":
            throw new InappropriateException();
        }
        throw new ParseException(file, reader.getLineNumber(), s + ": unknown type");
      case WORD:
        {
          var type = types.get(s);
          if (type != null) return type;
          type = new Func(null, s);
          types.put(s, type);
          return type;
        }
      default:
        throw new ParseException(file, reader.getLineNumber(), "type expected");
    }
  }

  private Object type() throws IOException {
    if (eat('(')) {
      var r = new ArrayList<>();
      r.add(null);
      do r.add(atomicType());
      while (eat('*'));
      expect(')');
      expect('>');
      var returnType = atomicType();
      r.set(0, returnType);
      return Array.ofAll(r);
    }
    var type = atomicType();
    if (eat('>')) {
      var returnType = atomicType();
      return Array.of(returnType, type);
    }
    return type;
  }

  private void args(Map<String, Variable> bound, ArrayList<Object> r) throws IOException {
    expect('(');
    do r.add(atomicTerm(bound));
    while (eat(','));
    expect(')');
  }

  private void args(Map<String, Variable> bound, ArrayList<Object> r, int arity)
      throws IOException {
    int n = r.size();
    args(bound, r);
    n = r.size() - n;
    if (n != arity)
      throw new ParseException(file, reader.getLineNumber(), "arg count: " + n + " != " + arity);
  }

  private Object definedAtomicTerm(Map<String, Variable> bound, Symbol op, int arity)
      throws IOException {
    var r = new ArrayList<>();
    r.add(op);
    args(bound, r, arity);
    return Array.of(r);
  }

  private Object atomicTerm(Map<String, Variable> bound) throws IOException {
    var k = token;
    var s = tokenString;
    lex();
    switch (k) {
      case '!':
      case '?':
      case '[':
        throw new InappropriateException();
      case DEFINED_WORD:
        switch (s) {
          case "$ceiling":
            return definedAtomicTerm(bound, Symbol.CEIL, 1);
          case "$difference":
            return definedAtomicTerm(bound, Symbol.SUBTRACT, 2);
          case "$distinct":
            {
              var r = new ArrayList<>();
              args(bound, r);
              var inequalities = new ArrayList<>();
              inequalities.add(Symbol.AND);
              for (var i = 0; i < r.size(); i++)
                for (var j = 0; j < r.size(); j++)
                  if (i != j)
                    inequalities.add(
                        Array.of(Symbol.NOT, Array.of(Symbol.EQUALS, r.get(i), r.get(j))));
              return Array.ofAll(inequalities);
            }
          case "$false":
            return false;
          case "$floor":
            return definedAtomicTerm(bound, Symbol.FLOOR, 1);
          case "$greater":
            {
              var r = new ArrayList<>();
              args(bound, r, 2);
              return Array.of(Symbol.LESS, r.get(1), r.get(0));
            }
          case "$greatereq":
            {
              var r = new ArrayList<>();
              args(bound, r, 2);
              return Array.of(Symbol.LESS_EQ, r.get(1), r.get(0));
            }
          case "$is_int":
            return definedAtomicTerm(bound, Symbol.IS_INTEGER, 1);
          case "$is_rat":
            return definedAtomicTerm(bound, Symbol.IS_RATIONAL, 1);
          case "$less":
            return definedAtomicTerm(bound, Symbol.LESS, 2);
          case "$lesseq":
            return definedAtomicTerm(bound, Symbol.LESS_EQ, 2);
          case "$product":
            return definedAtomicTerm(bound, Symbol.MULTIPLY, 2);
          case "$quotient":
            return definedAtomicTerm(bound, Symbol.DIVIDE, 2);
          case "$quotient_e":
            return definedAtomicTerm(bound, Symbol.DIVIDE_EUCLIDEAN, 2);
          case "$quotient_f":
            return definedAtomicTerm(bound, Symbol.DIVIDE_FLOOR, 2);
          case "$quotient_t":
            return definedAtomicTerm(bound, Symbol.DIVIDE_TRUNCATE, 2);
          case "$remainder_e":
            return definedAtomicTerm(bound, Symbol.REMAINDER_EUCLIDEAN, 2);
          case "$remainder_f":
            return definedAtomicTerm(bound, Symbol.REMAINDER_FLOOR, 2);
          case "$remainder_t":
            return definedAtomicTerm(bound, Symbol.REMAINDER_TRUNCATE, 2);
          case "$round":
            return definedAtomicTerm(bound, Symbol.ROUND, 1);
          case "$sum":
            return definedAtomicTerm(bound, Symbol.ADD, 2);
          case "$to_int":
            return definedAtomicTerm(bound, Symbol.TO_INTEGER, 1);
          case "$to_rat":
            return definedAtomicTerm(bound, Symbol.TO_RATIONAL, 1);
          case "$to_real":
            return definedAtomicTerm(bound, Symbol.TO_REAL, 1);
          case "$true":
            return true;
          case "$truncate":
            return definedAtomicTerm(bound, Symbol.TRUNCATE, 1);
          case "$uminus":
            return definedAtomicTerm(bound, Symbol.NEGATE, 1);
          default:
            throw new ParseException(file, reader.getLineNumber(), s + ": unknown word");
        }
      case DISTINCT_OBJECT:
        return s;
      case VARIABLE:
        {
          var a = free.get(s);
          if (a != null) return a;
          a = new Variable(Symbol.INDIVIDUAL);
          free.put(s, a);
          return a;
        }
      case WORD:
        {
          var a = funcs.get(s);
          if (a == null) {
            a = new Func(null, s);
            funcs.put(s, a);
          }
          if (token == '(') {
            var r = new ArrayList<>();
            r.add(a);
            args(bound, r);
            if (a.type == null) {
              var type = new Object[r.size()];
              for (var i = 0; i < type.length; i++) type[i] = new Variable(null);
              a.type = Array.of(type);
            }
            return Array.ofAll(r);
          }
          if (a.type == null) a.type = new Variable(null);
          return a;
        }
      case INTEGER:
        return new BigInteger(s);
      case RATIONAL:
        return BigRational.of(s);
      case REAL:
        // Real numbers are a problem
        // In general, they are incomputable
        // For computation purposes, double precision floating point is the best available
        // approximation
        // However, theorem proving needs exactness
        // So represent real number literals not as the usual floating point
        // but as 'the real number that would correspond to this rational number'
        return Array.of(Symbol.TO_REAL, BigRational.ofDecimal(s));
    }
    throw new ParseException(file, reader.getLineNumber(), ": term expected");
  }

  private Object infixUnary(Map<String, Variable> bound) throws IOException {
    var a = atomicTerm(bound);
    switch (token) {
      case '=':
        lex();
        return Array.of(Symbol.EQUALS, a, atomicTerm(bound));
      case NOT_EQ:
        lex();
        return Array.of(Symbol.NOT, Array.of(Symbol.EQUALS, a, atomicTerm(bound)));
      default:
        return a;
    }
  }

  private Object unaryFormulaBind(Map<String, Variable> bound, Symbol op) throws IOException {
    lex();
    expect('[');
    var params = new ArrayList<>();
    do {
      if (token != VARIABLE)
        throw new ParseException(file, reader.getLineNumber(), "variable expected");
      var name = tokenString;
      lex();
      var type = eat(':') ? null : Symbol.INDIVIDUAL;
      var a = new Variable(type);
      bound = bound.put(name, a);
      params.add(a);
    } while (eat(','));
    expect(']');
    expect(':');
    return Array.of(op, Array.ofAll(params), unaryFormula(bound));
  }

  private Object unaryFormula(Map<String, Variable> bound) throws IOException {
    switch (token) {
      case '!':
        return unaryFormulaBind(bound, Symbol.ALL);
      case '(':
        {
          lex();
          var a = logicFormula(bound);
          expect(')');
          return a;
        }
      case '?':
        return unaryFormulaBind(bound, Symbol.EXISTS);
      case '~':
        lex();
        return Array.of(Symbol.NOT, unaryFormula(bound));
    }
    return infixUnary(bound);
  }

  private Object logicFormulaRest(Map<String, Variable> bound, Symbol op, Object a)
      throws IOException {
    var k = token;
    var r = new ArrayList<>();
    r.add(op);
    r.add(a);
    while (eat(k)) r.add(unaryFormula(bound));
    return Array.ofAll(r);
  }

  private Object logicFormula(Map<String, Variable> bound) throws IOException {
    var a = unaryFormula(bound);
    switch (token) {
      case '&':
        return logicFormulaRest(bound, Symbol.AND, a);
      case '|':
        return logicFormulaRest(bound, Symbol.OR, a);
      case EQV:
        lex();
        return Array.of(Symbol.EQV, a, unaryFormula(bound));
      case IMPLIES:
        lex();
        return Etc.implies(a, unaryFormula(bound));
      case IMPLIESR:
        lex();
        return Etc.implies(unaryFormula(bound), a);
      case NAND:
        lex();
        return Array.of(Symbol.NOT, Array.of(Symbol.AND, a, unaryFormula(bound)));
      case NOR:
        lex();
        return Array.of(Symbol.NOT, Array.of(Symbol.OR, a, unaryFormula(bound)));
      case XOR:
        lex();
        return Array.of(Symbol.NOT, Array.of(Symbol.EQV, a, unaryFormula(bound)));
      default:
        return a;
    }
  }

  // Top level
  private String word() throws IOException {
    if (token != WORD) throw new ParseException(file, reader.getLineNumber(), "word expected");
    var s = tokenString;
    lex();
    return s;
  }

  private Object name() throws IOException {
    Object a;
    switch (token) {
      case WORD:
        a = tokenString;
        break;
      case INTEGER:
        a = Long.valueOf(tokenString);
        break;
      default:
        throw new ParseException(file, reader.getLineNumber(), "name expected");
    }
    lex();
    return a;
  }

  private void ignore() throws IOException {
    switch (token) {
      case '(':
        lex();
        while (!eat(')')) ignore();
        break;
      case -1:
        throw new ParseException(file, reader.getLineNumber(), "unexpected end of file");
      default:
        lex();
        break;
    }
  }

  private TptpParser(String file, InputStream stream, HashSet<Object> select) throws IOException {
    this.file = file;
    reader = new LineNumberReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
    reader.setLineNumber(1);
    c = reader.read();
    header = true;
    lex();
    header = false;
    if (!problem.header.isEmpty() && !problem.header.get(problem.header.size() - 1).isBlank())
      problem.header.add("");
    while (token != -1) {
      var s = word();
      expect('(');
      var name = name();
      switch (s) {
        case "cnf":
          {
            expect(',');

            // Role
            word();
            expect(',');

            // Literals
            free.clear();
            var negative = new ArrayList<>();
            var positive = new ArrayList<>();
            var parens = eat('(');
            do {
              var not = eat('~');
              var a = infixUnary(HashMap.empty());
              if (a instanceof Seq) {
                var a1 = (Seq) a;
                if (a1.head() == Symbol.NOT) {
                  a = a1.get(1);
                  not = !not;
                }
              }
              (not ? negative : positive).add(a);
            } while (eat('|'));
            if (parens) expect(')');
            if (select != null && !select.contains(name)) break;
            var c = new Clause(negative, positive, Inference.AXIOM);
            c.file = file;
            c.name = name;
            problem.clauses.add(c);
            break;
          }
        case "fof":
        case "tff":
          {
            expect(',');

            // Role
            var role = word();
            expect(',');

            // Term
            switch (role) {
              case "assumption":
              case "plain":
              case "unknown":
              case "axiom":
              case "corollary":
              case "definition":
              case "hypothesis":
              case "lemma":
              case "negated_conjecture":
              case "theorem":
                {
                  var a = logicFormula(HashMap.empty());
                  if (select != null && !select.contains(name)) break;
                  var formula = new Formula(a, Inference.AXIOM);
                  formula.file = file;
                  formula.name = name;
                  problem.formulas.add(formula);
                  break;
                }
              case "conjecture":
                {
                  var a = logicFormula(HashMap.empty());
                  if (select != null && !select.contains(name)) break;
                  if (problem.conjecture != null)
                    throw new ParseException(
                        file, reader.getLineNumber(), "multiple conjectures not supported");
                  var formula = new Formula(a, Inference.CONJECTURE);
                  formula.file = file;
                  formula.name = name;
                  problem.conjecture = formula;
                  break;
                }
              case "type":
                {
                  var parens = 0;
                  while (eat('(')) parens++;
                  var funcName = word();
                  expect(':');
                  if (token == DEFINED_WORD && tokenString.equals("$tType")) {
                    lex();
                    if (token == '>') throw new InappropriateException();
                  } else {
                    var type = type();
                    var a = funcs.get(funcName);
                    if (a == null) {
                      a = new Func(type, funcName);
                      funcs.put(funcName, a);
                    } else if (!Types.typeof(a).equals(type))
                      throw new ParseException(file, reader.getLineNumber(), "type mismatch");
                  }
                  while (parens-- > 0) expect(')');
                  break;
                }
              default:
                throw new ParseException(file, reader.getLineNumber(), role + ": unknown role");
            }
            break;
          }
        case "include":
          {
            // TPTP directory
            var tptp = System.getenv("TPTP");
            if (tptp == null) throw new IllegalStateException("TPTP environment variable not set");

            // File
            var file1 = tptp + '/' + name;
            var stream1 = new FileInputStream(file1);

            // Select
            var select1 = select;
            if (eat(','))
              if (token == WORD && "all".equals(tokenString)) lex();
              else {
                expect('[');
                select1 = new HashSet<>();
                do {
                  var name1 = word();
                  if (select == null || select.contains(name1)) select1.add(name1);
                } while (eat(','));
                expect(']');
              }

            // Read
            new TptpParser(file1, stream1, select1);
            break;
          }
        case "thf":
          throw new InappropriateException();
        default:
          throw new ParseException(file, reader.getLineNumber(), "unknown language");
      }
      if (token == ',') do ignore(); while (token != ')');
      expect(')');
      expect('.');
    }
  }

  public static Problem read(String file, InputStream stream) throws IOException {
    funcs = new java.util.HashMap<>();
    problem = new Problem(file);

    // Read
    new TptpParser(file, stream, null);
    Types.inferTypes(problem.formulas, problem.clauses);

    // Free memory
    funcs = null;

    // Return
    return problem;
  }
}
