package prover;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;

public final class TptpParser {
  private static final Pattern STATUS_PATTERN = Pattern.compile("\\s*Status\\s*:\\s*(\\w+)");

  // Tokens
  private static final int NOT_EQ = -2;
  private static final int VAR = -3;
  private static final int WORD = -4;

  // Problem state
  private static List<Clause> clauses;
  private static Map<String, Function> functions;

  // File state
  private final String file;
  private final LineNumberReader reader;
  private int c;
  private int tok;
  private String tokString;
  private Map<String, Variable> free = new HashMap<>();

  @SuppressWarnings("fallthrough")
  private TptpParser(String file, Set<String> select) throws IOException {
    this.file = file;
    reader =
        new LineNumberReader(
            new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8));
    reader.setLineNumber(1);
    c = reader.read();
    lex();
    while (tok != -1) {
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
            var negative = new ArrayList<Term>();
            var positive = new ArrayList<Term>();
            var parens = eat('(');
            do {
              var not = eat('~');
              var a = infixUnary();
              if (a instanceof Not) {
                not = !not;
                a = a.get(0);
              }
              (not ? negative : positive).add(a);
            } while (eat('|'));
            if (parens) {
              expect(')');
            }
            if ((select != null) && !select.contains(name)) {
              break;
            }
            clauses.add(new Clause(negative, positive));
            break;
          }
        case "include":
          {
            var tptp = System.getenv("TPTP");
            if (tptp == null) {
              throw new IllegalStateException("TPTP environment variable not set");
            }
            var select1 = select;
            if (eat(',')) {
              if ((tok == WORD) && "all".equals(tokString)) {
                lex();
              } else {
                expect('[');
                select1 = new HashSet<>();
                do {
                  var name1 = word();
                  if ((select == null) || select.contains(name1)) {
                    select1.add(name1);
                  }
                } while (eat(','));
                expect(']');
              }
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

  public static List<Clause> read(String file) throws IOException {
    clauses = new ArrayList<>();
    functions = new HashMap<>();
    new TptpParser(file, null);
    return clauses;
  }

  private void args(List<Term> r) throws IOException {
    expect('(');
    do {
      r.add(atomicTerm());
    } while (eat(','));
    expect(')');
  }

  private Term atomicTerm() throws IOException {
    var k = tok;
    var s = tokString;
    lex();
    switch (k) {
      case VAR:
        {
          var a = free.get(s);
          if (a != null) {
            return a;
          }
          a = new Variable(Type.INDIVIDUAL, s);
          free.put(s, a);
          return a;
        }
      case WORD:
        {
          var a = functions.get(s);
          if (a == null) {
            a = new Function(s);
            functions.put(s, a);
          }
          if (tok == '(') {
            var r = new ArrayList<Term>();
            args(r);
            return a.call(r);
          }
          return a;
        }
      default:
        throw new ParseException(file, reader.getLineNumber(), ": term expected");
    }
  }

  private boolean eat(int k) throws IOException {
    if (tok == k) {
      lex();
      return true;
    }
    return false;
  }

  private void expect(int k) throws IOException {
    if (eat(k)) {
      return;
    }
    throw new ParseException(file, reader.getLineNumber(), ": '" + (char) k + "' expected");
  }

  private Term infixUnary() throws IOException {
    var a = atomicTerm();
    switch (tok) {
      case '=':
        lex();
        return new Eq(a, atomicTerm());
      case NOT_EQ:
        lex();
        return new Not(new Eq(a, atomicTerm()));
      default:
        return a;
    }
  }

  private void lex() throws IOException {
    for (; ; ) {
      tok = c;
      switch (c) {
        case ' ':
        case '\f':
        case '\n':
        case '\r':
        case '\t':
          c = reader.read();
          continue;
        case '!':
          c = reader.read();
          if (c == '=') {
            c = reader.read();
            tok = NOT_EQ;
            break;
          }
          break;
        case '%':
          var s = reader.readLine();
          c = reader.read();
          if (Main.status == null) {
            var matcher = STATUS_PATTERN.matcher(s);
            if (matcher.matches()) {
              switch (matcher.group(1)) {
                case "Satisfiable":
                  Main.status = true;
                  break;
                case "Unsatisfiable":
                  Main.status = false;
                  break;
              }
            }
          }
          continue;
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
          {
            var sb = new StringBuilder();
            do {
              sb.append((char) c);
              c = reader.read();
            } while (Character.isJavaIdentifierPart(c));
            tok = VAR;
            tokString = sb.toString();
            break;
          }
        case '\'':
          var line = reader.getLineNumber();
          var quote = c;
          var sb1 = new StringBuilder();
          c = reader.read();
          while (c != quote) {
            if (c < ' ') {
              throw new ParseException(file, line, "unclosed quote");
            }
            if (c == '\\') {
              c = reader.read();
            }
            sb1.append((char) c);
            c = reader.read();
          }
          c = reader.read();
          tokString = sb1.toString();
          tok = WORD;
          break;
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
          {
            var sb = new StringBuilder();
            do {
              sb.append((char) c);
              c = reader.read();
            } while (Character.isJavaIdentifierPart(c));
            tok = WORD;
            tokString = sb.toString();
            break;
          }
        default:
          c = reader.read();
          break;
      }
      return;
    }
  }

  private String word() throws IOException {
    if (tok != WORD) {
      throw new ParseException(file, reader.getLineNumber(), "word expected");
    }
    var s = tokString;
    lex();
    return s;
  }
}
