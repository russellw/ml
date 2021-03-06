package prover;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;

public final class DimacsParser {
  private static final Pattern STATUS_PATTERN = Pattern.compile(".* (SAT|UNSAT) .*");

  private static final int INTEGER = -2;
  private static final int ZERO = -3;

  private final Problem problem;

  private final LineNumberReader reader;
  private int c;
  private int token;
  private String tokenString;

  private List<Object> negative = new ArrayList<>();
  private List<Object> positive = new ArrayList<>();

  // Tokenizer
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
        case '0':
          c = reader.read();
          token = ZERO;
          return;
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
            } while ('0' <= c && c <= '9');
            token = INTEGER;
            tokenString = sb.toString();
            return;
          }
        case 'c':
          {
            var s = reader.readLine();
            c = reader.read();
            System.out.println(s);
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
          }
      }
      c = reader.read();
      return;
    }
  }

  // Variables in propositional logic are functions in first-order logic
  // Though DIMACS uses propositional terminology,
  // we use first-order terminology everywhere for consistency
  private Object func() throws IOException {
    var a = problem.funcs.get(tokenString);
    if (a == null) {
      a = new Func(Symbol.BOOLEAN, tokenString);
      problem.funcs.put(tokenString, a);
    }
    lex();
    return a;
  }

  // Top level
  private void clause() {
    var c = new Clause(negative, positive, Inference.AXIOM);
    c.file = problem.file;
    problem.clauses.add(c);
  }

  private DimacsParser(Problem problem, InputStream stream) throws IOException {
    this.problem = problem;
    reader = new LineNumberReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
    reader.setLineNumber(1);
    c = reader.read();
    lex();

    // Problem statistics
    if (token == 'p') {
      while (Character.isWhitespace(c)) c = reader.read();

      // cnf
      if (c != 'c')
        throw new ParseException(problem.file, reader.getLineNumber(), "'cnf' expected");
      c = reader.read();
      if (c != 'n')
        throw new ParseException(problem.file, reader.getLineNumber(), "'cnf' expected");
      c = reader.read();
      if (c != 'f')
        throw new ParseException(problem.file, reader.getLineNumber(), "'cnf' expected");
      c = reader.read();
      lex();

      // Variables
      if (token != INTEGER)
        throw new ParseException(problem.file, reader.getLineNumber(), "count expected");
      lex();

      // Clauses
      if (token != INTEGER)
        throw new ParseException(problem.file, reader.getLineNumber(), "count expected");
      lex();
    }

    // Clauses
    for (; ; )
      switch (token) {
        case '-':
          lex();
          negative.add(func());
          break;
        case -1:
          if (negative.size() + positive.size() > 0) clause();
          return;
        case INTEGER:
          positive.add(func());
          break;
        case ZERO:
          lex();
          clause();
          negative.clear();
          positive.clear();
          break;
        default:
          throw new ParseException(problem.file, reader.getLineNumber(), "syntax error");
      }
  }

  public static void read(Problem problem, InputStream stream) throws IOException {
    new DimacsParser(problem, stream);
  }
}
