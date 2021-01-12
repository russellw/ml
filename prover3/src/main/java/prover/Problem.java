package prover;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.NumberFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public final class Problem {
  public final String file;
  public List<String> header = new ArrayList<>();
  public SZS expected;
  public List<Formula> formulas = new ArrayList<>();
  public Formula conjecture;
  public List<Clause> clauses = new ArrayList<>();
  public Clause refutation;
  public SZS result;

  public void solve(long deadline) {
    if (conjecture != null)
      formulas.add(
          new Formula(List.of(Symbol.NOT, conjecture.term()), Inference.NEGATE, conjecture));
    Types.inferTypes(formulas, clauses);
    CNF.convert(formulas, clauses);
    Superposition.solve(this, deadline);
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

  public Problem(String file) {
    this.file = file;
  }

  public void write() throws IOException {
    // Report
    new File("logs").mkdir();
    var writer = new PrintWriter("logs/" + Etc.removeExtension(Etc.removeDir(file)) + ".html");
    var numberFormat = NumberFormat.getInstance();

    // Header
    writer.println("<!DOCTYPE html>");
    writer.println("<html lang=\"en\">");
    writer.println("<meta charset=\"utf-8\"/>");
    writer.printf("<title>%s</title>\n", file);
    writer.println("<style>");
    writer.println("h1 {");
    writer.println("font-size: 150%;");
    writer.println("}");
    writer.println("h2 {");
    writer.println("font-size: 125%;");
    writer.println("}");
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

    // Contents
    writer.println("<h1 id=\"Contents\">Contents</h1>");
    writer.println("<ul>");
    writer.println("<li><a href=\"#Contents\">Contents</a>");
    writer.println("<li><a href=\"#Input-files\">Input files</a>");
    writer.println("<li><a href=\"#Clauses\">Clauses</a>");
    writer.println("<li><a href=\"#Subsumption\">Subsumption</a>");
    writer.println("<li><a href=\"#Result\">Result</a>");
    if (refutation != null) writer.println("<li><a href=\"#Proof\">Proof</a>");
    writer.println("<li><a href=\"#Memory\">Memory</a>");
    writer.println("<li><a href=\"#Time\">Time</a>");
    writer.println("</ul>");

    // Problem header
    if (!header.isEmpty()) {
      if (header.get(header.size() - 1).isEmpty()) {
        header.remove(header.size() - 1);
      }
      writer.println("<h1 id=\"Problem-header\">Problem header</h1>");
      writer.println("<pre>");
      for (var s : header) {
        wrap(s, writer);
      }
      writer.println("</pre>");
    }

    // Result
    writer.println("<h1 id=\"Result\">Result</h1>");
    writer.println("<table class=\"bordered\">");
    if (expected != null) {
      writer.println("<tr>");
      writer.println("<td class=\"bordered\">Expected");
      writer.println("<td class=\"bordered\">" + expected);
    }
    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Result");
    writer.printf("<td class=\"bordered\"><b>%s</b>\n", result);
    writer.println("</table>");

    // Proof
    if (refutation != null) {
      writer.println("<h1 id=\"Proof\">Proof</h1>");
      writer.println("<code>");
      writer.println("</code>");
    }

    // Memory
    var runtime = Runtime.getRuntime();
    writer.println("<h1 id=\"Memory\">Memory</h1>");
    writer.println("<table class=\"bordered\">");
    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Current");
    writer.println(
        "<td class=\"bordered\"; style=\"text-align: right\">"
            + numberFormat.format(runtime.totalMemory() - runtime.freeMemory()));
    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Free");
    writer.println(
        "<td class=\"bordered\"; style=\"text-align: right\">+ "
            + numberFormat.format(runtime.freeMemory()));
    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Total");
    writer.println(
        "<td class=\"bordered\"; style=\"text-align: right\">= "
            + numberFormat.format(runtime.totalMemory()));
    writer.println("<tr>");
    writer.println("<td class=\"bordered\">Max");
    writer.println(
        "<td class=\"bordered\"; style=\"text-align: right\">"
            + numberFormat.format(runtime.maxMemory()));
    writer.println("</table>");

    // Time
    writer.println("<h1 id=\"Time\">Time</h1>");
    writer.println(
        "<p>"
            + LocalDateTime.now()
                .format(DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy, HH:mm:ss")));

    // Flush output
    writer.close();
  }

  private static void wrap(String s, PrintWriter writer) {
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
