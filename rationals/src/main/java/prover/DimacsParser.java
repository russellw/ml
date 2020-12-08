package prover;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Pattern;

public final class DimacsParser {
  private static final Pattern STATUS_PATTERN = Pattern.compile(".* (SAT|UNSAT) .*");

  // Tokens
  private static final int INTEGER = -2;
  private static final int ZERO = -3;

  // Problem state
  private final Problem problem = new Problem();
  private final Map<String, Function> functions = new HashMap<>();

  // File state
  private final LineNumberReader reader;
  private int c;
  private int tok;
  private String tokString;

  private DimacsParser(Path path, InputStream stream) throws IOException {
    var file = path.getFileName().toString();
    reader = new LineNumberReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
    reader.setLineNumber(1);
    c = reader.read();
    lex();

    // Problem statistics
    if (tok == 'p') {
      while (Character.isWhitespace(c)) {
        c = reader.read();
      }

      // cnf
      if (c != 'c') {
        throw new ParseException(file, reader.getLineNumber(), "'cnf' expected");
      }
      c = reader.read();
      if (c != 'n') {
        throw new ParseException(file, reader.getLineNumber(), "'cnf' expected");
      }
      c = reader.read();
      if (c != 'f') {
        throw new ParseException(file, reader.getLineNumber(), "'cnf' expected");
      }
      c = reader.read();
      lex();

      // Variables
      if (tok != INTEGER) {
        throw new ParseException(file, reader.getLineNumber(), "count expected");
      }
      lex();

      // Clauses
      if (tok != INTEGER) {
        throw new ParseException(file, reader.getLineNumber(), "count expected");
      }
      lex();
    }

    // Clauses
    var negative = new ArrayList<Term>();
    var positive = new ArrayList<Term>();
    for (; ; ) {
      switch (tok) {
        case '-':
          lex();
          negative.add(function());
          break;
        case -1:
          if (negative.size() + positive.size() > 0) {
            problem.clauses.add(new ClauseInput(negative, positive, null, file));
          }
          return;
        case INTEGER:
          positive.add(function());
          break;
        case ZERO:
          lex();
          problem.clauses.add(new ClauseInput(negative, positive, null, file));
          negative.clear();
          positive.clear();
          break;
        default:
          throw new ParseException(file, reader.getLineNumber(), "syntax error");
      }
    }
  }

  // Variables in propositional logic are functions in first-order logic
  // Though DIMACS uses propositional terminology,
  // we use first-order terminology everywhere for consistency
  private Term function() throws IOException {
    var a = functions.get(tokString);
    if (a == null) {
      a = new Function(Type.BOOLEAN, tokString);
      functions.put(tokString, a);
    }
    lex();
    return a;
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
        case '0':
          c = reader.read();
          tok = ZERO;
          break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
          var sb = new StringBuilder();
          do {
            sb.append((char) c);
            c = reader.read();
          } while (('0' <= c) && (c <= '9'));
          tok = INTEGER;
          tokString = sb.toString();
          break;
        case 'c':
          var s = reader.readLine();
          c = reader.read();
          if (problem.expected == null) {
            var matcher = STATUS_PATTERN.matcher(s);
            if (matcher.matches()) {
              switch (matcher.group(1)) {
                case "SAT":
                  problem.expected = SZS.Satisfiable;
                  break;
                case "UNSAT":
                  problem.expected = SZS.Unsatisfiable;
                  break;
              }
            }
          }
          continue;
        default:
          c = reader.read();
          break;
      }
      return;
    }
  }

  public static Problem read(Path path, InputStream stream) throws IOException {
    return new DimacsParser(path, stream).problem;
  }
}
