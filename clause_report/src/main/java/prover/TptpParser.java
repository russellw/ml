package prover;

import io.vavr.collection.Array;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
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
  private static Problem problem;
  private static HashMap<String, Func> functions;

  // File state
  private final String file;
  private final LineNumberReader reader;
  private int c;
  private int token;
  private String tokenString;
  private HashMap<String, Variable> free = new HashMap<>();

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

  private void args(ArrayList<Object> r) throws IOException {
    expect('(');
    do r.add(atomicTerm());
    while (eat(','));
    expect(')');
  }

  private Object atomicTerm() throws IOException {
    var k = token;
    var s = tokenString;
    lex();
    switch (k) {
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
          var a = functions.get(s);
          if (a == null) {
            a = new Func(s);
            functions.put(s, a);
          }
          if (token == '(') {
            var r = new ArrayList<>();
            r.add(a);
            args(r);
            return Array.of(r.toArray());
          }
          return a;
        }
      default:
        throw new ParseException(file, reader.getLineNumber(), ": term expected");
    }
  }

  private Object infixUnary() throws IOException {
    var a = atomicTerm();
    switch (token) {
      case '=':
        lex();
        return Array.of(Symbol.EQUALS, a, atomicTerm());
      case NOT_EQ:
        lex();
        return new Not(Array.of(Symbol.EQUALS, a, atomicTerm()));
      default:
        return a;
    }
  }

  private String word() throws IOException {
    if (token != WORD) throw new ParseException(file, reader.getLineNumber(), "word expected");
    var s = tokenString;
    lex();
    return s;
  }

  private TptpParser(String file, HashSet<String> select) throws IOException {
    this.file = file;
    reader =
        new LineNumberReader(
            new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8));
    reader.setLineNumber(1);
    c = reader.read();
    lex();
    while (token != -1) {
      var s = word();
      expect('(');
      var name = word();
      switch (s) {
        case "cnf":
          {
            expect(',');

            // Role
            word();
            expect(',');

            // Formula
            free.clear();
            var negative = new ArrayList<>();
            var positive = new ArrayList<>();
            var parens = eat('(');
            do {
              var not = eat('~');
              var a = infixUnary();
              if (a instanceof Not) {
                a = ((Not) a).a;
                not = !not;
              }
              (not ? negative : positive).add(a);
            } while (eat('|'));
            if (parens) expect(')');
            if (select != null && !select.contains(name)) break;
            problem.clauses.add(new Clause(negative, positive));
            break;
          }
        case "include":
          {
            var tptp = System.getenv("TPTP");
            if (tptp == null) throw new IllegalStateException("TPTP environment variable not set");
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
            new TptpParser(tptp + '/' + name, select1);
            break;
          }
        default:
          throw new ParseException(file, reader.getLineNumber(), "unknown language");
      }
      expect(')');
      expect('.');
    }
  }

  public static Problem read(String file) throws IOException {
    functions = new HashMap<>();
    problem = new Problem();

    // Read
    new TptpParser(file, null);

    // Free memory
    functions = null;

    // Return
    return problem;
  }

  private static final class Not {
    final Object a;

    private Not(Object a) {
      this.a = a;
    }
  }
}
