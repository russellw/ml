package prover;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public final class Problem {
  public final long startTime = System.currentTimeMillis();
  public long endTime;
  private final List<String> files = new ArrayList<>();
  public final List<String> header = new ArrayList<>();
  public SZS expected;
  public double rating = -1;
  public final List<Formula> formulas = new ArrayList<>();
  public Formula conjecture;
  public final Set<Func> skolems = new LinkedHashSet<>();
  public final List<Clause> clauses = new ArrayList<>();
  public Superposition superposition;
  public Clause refutation;
  public SZS result;
  List<Record> records = new ArrayList<>();

  // Statistics
  public long timeParser;
  private long timeTypeInference;
  private long timeCnfConversion;
  private long timeSuperposition;

  // Output
  private PrintWriter writer;

  private void term(Object a) {
    if (a instanceof List) {
      var a1 = (List) a;
      writer.print('(');
      for (var i = 0; i < a1.size(); i++) {
        if (i > 0) writer.print(' ');
        term(a1.get(i));
      }
      writer.print(')');
      return;
    }
    writer.print(a);
  }

  public String file() {
    return files.get(0);
  }

  public void add(String file, int includeDepth) {
    files.add("\t".repeat(includeDepth) + file);
  }

  public void solve(long timeout) {
    if (conjecture != null)
      formulas.add(
          new Formula(List.of(Symbol.NOT, conjecture.term()), Inference.NEGATE, conjecture));
    timeTypeInference = Etc.time(() -> Types.inferTypes(formulas, clauses));
    timeCnfConversion = Etc.time(() -> new CNF(this));
    timeSuperposition =
        Etc.time(
            () -> {
              superposition = new Superposition();
              superposition.solve(this, startTime + timeout);
            });
    if (conjecture != null)
      switch (result) {
        case Satisfiable:
          result = SZS.CounterSatisfiable;
          break;
        case Unsatisfiable:
          result = SZS.Theorem;
          break;
      }
    if (expected != null && result != expected)
      switch (result) {
        case Unsatisfiable:
        case Theorem:
          if (expected == SZS.ContradictoryAxioms) break;
        case Satisfiable:
        case CounterSatisfiable:
          throw new IllegalStateException(result + " != " + expected);
      }
    endTime = System.currentTimeMillis();
  }

  private void func(Func a, Integer n) {
    writer.println("<tr>");

    writer.print("<td class=\"bordered\"><code>");
    writer.print(a);
    writer.println("</code>");

    writer.print("<td class=\"bordered\"><code>");
    writer.print(Types.typeof(a));
    writer.println("</code>");

    writer.print("<td class=\"bordered\" style=\"text-align: right\">");
    if (n != null) writer.print(n);
    writer.println();
  }

  public void write() throws IOException {}

  public static boolean solved(SZS szs) {
    switch (szs) {
      case Unsatisfiable:
      case ContradictoryAxioms:
      case Theorem:
      case CounterSatisfiable:
      case Satisfiable:
        return true;
    }
    return false;
  }

  private void wrap(String s) {
    var column = 0;
    for (var i = 0; i < s.length(); ) {
      var j = i;

      // Find start of next word
      while (j < s.length() && s.charAt(j) == ' ') j++;
      var word = j;

      // Find end of next word, within length limit
      while (j < s.length() && s.charAt(j) != ' ') j++;
      j = Math.min(j, word + 90);

      // If printing that much would take us past 80 columns
      // (unless we are already at the start of a line)
      if (column > 0 && column + j - i > 80) {
        // Skip leading space
        while (i < j && s.charAt(i) == ' ') i++;

        // New line
        writer.println();
        column = 0;
      }

      // Print word
      var t = s.substring(i, j).replace("<", "&lt;");
      writer.print(t);
      column += t.length();
      i = j;
    }
    writer.println();
  }
}
