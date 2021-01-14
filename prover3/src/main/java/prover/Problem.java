package prover;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.text.NumberFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public final class Problem {
  public final long start = System.currentTimeMillis();
  private final List<String> files = new ArrayList<>();
  public final List<String> header = new ArrayList<>();
  public SZS expected;
  public final List<Formula> formulas = new ArrayList<>();
  public Formula conjecture;
  public final Set<Func> skolems = new LinkedHashSet<>();
  public final List<Clause> clauses = new ArrayList<>();
  public Clause refutation;
  public SZS result;

  // Statistics
  public long timeParser;
  private long timeTypeInference;
  private long timeCnfConversion;
  private long timeSuperposition;

  // Output
  private PrintWriter writer;

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
    timeCnfConversion = Etc.time(() -> CNF.convert(this));
    timeSuperposition = Etc.time(() -> Superposition.solve(this, start + timeout));
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
  }

  private void href(String s) {
    s = Path.of(s).toAbsolutePath().toString();
    writer.printf("<a href=\"%s\">%s</a>", s.replace('\\', '/'), s);
  }

  private void func(Func a, int n) {
    writer.println("<tr>");
    writer.print("<td class=\"bordered\">");
    writer.println(a);
    writer.print("<td class=\"bordered\">");
    writer.println(Types.typeof(a));
    writer.print("<td class=\"bordered\" style=\"text-align: right\">");
    writer.println(n);
  }

  public void write() throws IOException {
    // Report
    var name = Etc.baseName(file());
    writer = new PrintWriter(Main.logDir + '/' + name + ".html");
    var numberFormat = NumberFormat.getInstance();

    // HTML header
    writer.println("<!DOCTYPE html>");
    writer.println("<html lang=\"en\">");
    writer.println("<meta charset=\"utf-8\"/>");
    writer.printf("<title>%s</title>\n", name);
    writer.println("<style>");
    writer.println("caption {");
    writer.println("text-align: left;");
    writer.println("white-space: nowrap;");
    writer.println("}");
    writer.println("table.bordered, th.bordered, td.bordered {");
    writer.println("border: 1px solid;");
    writer.println("border-collapse: collapse;");
    writer.println("padding: 5px;");
    writer.println("}");
    writer.println("table.padded, th.padded, td.padded {");
    writer.println("padding: 3px;");
    writer.println("}");
    writer.println("td.fixed {");
    writer.println("white-space: nowrap;");
    writer.println("}");
    writer.println("td.bar {");
    writer.println("width: 100%");
    writer.println("}");
    writer.println("</style>");

    // Problem header
    while (!header.isEmpty() && header.get(header.size() - 1).isBlank())
      header.remove(header.size() - 1);

    // Operators
    var ops = new Bag<>();
    for (var c : clauses)
      Etc.walkBranches(
          Arrays.asList(c.literals),
          a -> {
            var op = a.get(0);
            if (op instanceof Symbol) ops.add(List.of(op, Types.typeof(a)));
          });

    // Contents
    writer.println("<h1 id=\"Contents\">Contents</h1>");
    writer.println("<ul>");
    writer.println("<li><a href=\"#Contents\">Contents</a>");
    writer.println("<li><a href=\"#Input-files\">Input files</a>");
    if (!header.isEmpty()) writer.println("<li><a href=\"#Problem-header\">Problem header</a>");
    if (!ops.isEmpty()) writer.println("<li><a href=\"#Operators\">Operators</a>");
    writer.println("<li><a href=\"#Functions\">Functions</a>");
    writer.println("<li><a href=\"#Subsumption\">Subsumption</a>");
    writer.println("<li><a href=\"#Result\">Result</a>");
    if (refutation != null) writer.println("<li><a href=\"#Proof\">Proof</a>");
    writer.println("<li><a href=\"#Memory\">Memory</a>");
    writer.println("<li><a href=\"#Time\">Time</a>");
    writer.println("</ul>");

    // Input files
    writer.println("<h1 id=\"Input-files\">Input files</h1>");
    var includeDepth = -1;
    for (var file : files) {
      var i = 0;
      while (file.charAt(i) == '\t') i++;
      file = file.substring(i);
      if (i > includeDepth) writer.println("<ul>");
      if (i < includeDepth) writer.println("</ul>");
      includeDepth = i;
      writer.print("<li>");
      href(file);
      writer.println();
    }
    for (var i = -1; i < includeDepth; i++) writer.println("</ul>");

    // Problem header
    if (!header.isEmpty()) {
      writer.println("<h1 id=\"Problem-header\">Problem header</h1>");
      writer.println("<pre>");
      for (var s : header) wrap(s);
      writer.println("</pre>");
    }

    // Operators
    writer.println("<h1 id=\"Operators\">Operators</h1>");
    writer.println("<table class=\"bordered\">");
    writer.println("<tr>");
    writer.println("<th class=\"bordered\">Name");
    writer.println("<th class=\"bordered\">Type");
    writer.println("<th class=\"bordered\">Occurs");
    var ops1 = new ArrayList<>(ops.keySet());
    ops1.sort(Comparator.comparing(Object::toString));
    for (var a : ops1) {
      var a1 = (List) a;
      var op = a1.get(0);
      var type = a1.get(1);
      writer.println("<tr>");
      writer.print("<td class=\"bordered\">");
      writer.println(op);
      writer.print("<td class=\"bordered\">");
      writer.println(type);
      writer.print("<td class=\"bordered\" style=\"text-align: right\">");
      writer.println(ops.get(a));
    }
    writer.println("</table>");

    // Functions
    writer.println("<h1 id=\"Functions\">Functions</h1>");
    writer.println("<table class=\"bordered\">");
    writer.println("<tr>");
    writer.println("<th class=\"bordered\">Name");
    writer.println("<th class=\"bordered\">Type");
    writer.println("<th class=\"bordered\">Occurs");
    var funcs = new Bag<Func>();
    for (var c : clauses)
      Etc.walkLeaves(
          Arrays.asList(c.literals),
          a -> {
            if (a instanceof Func) funcs.add((Func) a);
          });
    var funcs1 = new ArrayList<>(funcs.keySet());
    funcs1.sort(Comparator.comparing(Func::toString));
    for (var a : funcs1) if (!skolems.contains(a)) func(a, funcs.get(a));
    if (!skolems.isEmpty()) {
      writer.println("<tr>");
      writer.println("<th class=\"bordered\" colspan=\"3\">Skolem functions");
      for (var a : skolems) func(a, funcs.get(a));
    }
    writer.println("</table>");

    // Result
    writer.println("<h1 id=\"Result\">Result</h1>");
    writer.println("<table class=\"bordered\">");
    if (expected != null) {
      writer.println("<tr>");
      writer.println("<td class=\"bordered\">Expected");
      writer.print("<td class=\"bordered\">");
      writer.println(expected);
    }
    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Result");
    writer.printf("<td class=\"bordered\"><b>%s</b>\n", result);
    writer.println("</table>");

    // Proof
    if (refutation != null) {
      writer.println("<h1 id=\"Proof\">Proof</h1>");
      writer.println("<table class=\"bordered\">");
      var proof = refutation.proof();
      if (!formulas.isEmpty()) {
        writer.println("<tr>");
        writer.println("<th class=\"bordered\">From");
        writer.println("<th class=\"bordered\">Inference");
        writer.println("<th class=\"bordered\">Name");
        writer.println("<th class=\"bordered\" colspan=\"2\">Term");
        for (var formula : proof) {
          if (!(formula instanceof Formula)) continue;
          writer.println("<tr>");

          writer.print("<td class=\"bordered\">");
          for (var i = 0; i < formula.from.length; i++) {
            if (i > 0) writer.print(' ');
            writer.print(formula.from[i].name);
          }
          writer.println();

          writer.print("<td class=\"bordered\">");
          writer.println(formula.inference);

          writer.print("<td class=\"bordered\">");
          writer.println(formula.name);

          writer.print("<td class=\"bordered\" colspan=\"2\">");
          writer.println(formula.term());
        }
      }
      writer.println("<tr>");
      writer.println("<th class=\"bordered\">From");
      writer.println("<th class=\"bordered\">Inference");
      writer.println("<th class=\"bordered\">Name");
      writer.println("<th class=\"bordered\">Negative");
      writer.println("<th class=\"bordered\">Positive");
      for (var formula : proof) {
        if (!(formula instanceof Clause)) continue;
        var c = (Clause) formula;
        writer.println("<tr>");

        writer.print("<td class=\"bordered\">");
        for (var i = 0; i < c.from.length; i++) {
          if (i > 0) writer.print(' ');
          writer.print(c.from[i].name);
        }
        writer.println();

        writer.print("<td class=\"bordered\">");
        writer.println(c.inference);

        writer.print("<td class=\"bordered\">");
        writer.println(c.name);

        writer.print("<td class=\"bordered\">");
        var negative = c.negative();
        for (var i = 0; i < negative.length; i++) {
          if (i > 0) writer.print(' ');
          writer.print(negative[i]);
        }
        writer.println();

        writer.print("<td class=\"bordered\">");
        var positive = c.positive();
        for (var i = 0; i < positive.length; i++) {
          if (i > 0) writer.print(' ');
          writer.print(positive[i]);
        }
        writer.println();
      }
      writer.println("</table>");
    }

    // Memory
    var runtime = Runtime.getRuntime();
    writer.println("<h1 id=\"Memory\">Memory</h1>");
    writer.println("<table class=\"bordered\">");

    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Current");
    writer.print("<td class=\"bordered\" style=\"text-align: right\">");
    writer.println(numberFormat.format(runtime.totalMemory() - runtime.freeMemory()));

    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Free");
    writer.print("<td class=\"bordered\" style=\"text-align: right\">+ ");
    writer.println(numberFormat.format(runtime.freeMemory()));

    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Total");
    writer.print("<td class=\"bordered\" style=\"text-align: right\">= ");
    writer.println(numberFormat.format(runtime.totalMemory()));

    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Max");
    writer.print("<td class=\"bordered\" style=\"text-align: right\">");
    writer.println(numberFormat.format(runtime.maxMemory()));

    writer.println("</table>");

    // Time
    writer.println("<h1 id=\"Time\">Time</h1>");
    writer.println("<table class=\"bordered\">");

    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Parser");
    writer.printf("<td class=\"bordered\" style=\"text-align: right\">%.3f\n", timeParser * 0.001);
    writer.println("</tr>");

    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Type inference");
    writer.printf(
        "<td class=\"bordered\" style=\"text-align: right\">%.3f\n", timeTypeInference * 0.001);
    writer.println("</tr>");

    writer.println("<tr>");
    writer.println("<td class=\"bordered\">CNF conversion");
    writer.printf(
        "<td class=\"bordered\" style=\"text-align: right\">%.3f\n", timeCnfConversion * 0.001);
    writer.println("</tr>");

    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Superposition");
    writer.printf(
        "<td class=\"bordered\" style=\"text-align: right\">%.3f\n", timeSuperposition * 0.001);
    writer.println("</tr>");

    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Total");
    writer.printf(
        "<td class=\"bordered\" style=\"text-align: right\">%.3f\n",
        (System.currentTimeMillis() - start) * 0.001);
    writer.println("</tr>");

    writer.println("</table>");

    writer.print("<p>");
    writer.println(
        LocalDateTime.now().format(DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy, HH:mm:ss")));

    // Flush output
    writer.close();
  }

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
