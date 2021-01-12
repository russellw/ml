package prover;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
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
    var out = new PrintWriter("logs/" + Etc.removeExtension(Etc.removeDir(file)) + ".html");
    out.println("<!DOCTYPE html>");
    out.println("<html lang=\"en\">");
    out.println("<meta charset=\"utf-8\"/>");
    out.printf("<title>%s</title>\n", file);
    out.println("<style>");
    out.println("h1 {");
    out.println("font-size: 150%;");
    out.println("}");
    out.println("h2 {");
    out.println("font-size: 125%;");
    out.println("}");
    out.println("caption {");
    out.println("text-align: left;");
    out.println("white-space: nowrap;");
    out.println("}");
    out.println("table.bordered, th.bordered, td.bordered {");
    out.println("border: 1px solid;");
    out.println("border-collapse: collapse;");
    out.println("padding: 5px;");
    out.println("}");
    out.println("table.padded, th.padded, td.padded {");
    out.println("padding: 3px;");
    out.println("}");
    out.println("td.fixed {");
    out.println("white-space: nowrap;");
    out.println("}");
    out.println("td.bar {");
    out.println("width: 100%");
    out.println("}");
    out.println("</style>");

    // Contents
    out.println("<h1 id=\"Contents\">Contents</h1>");
    out.println("<ul>");
    out.println("<li><a href=\"#Contents\">Contents</a>");
    out.println("<li><a href=\"#Input-files\">Input files</a>");
    out.println("<li><a href=\"#Clauses\">Clauses</a>");
    out.println("<li><a href=\"#Subsumption\">Subsumption</a>");
    out.println("<li><a href=\"#Result\">Result</a>");
    if (refutation != null) out.println("<li><a href=\"#Proof\">Proof</a>");
    out.println("<li><a href=\"#Memory\">Memory</a>");
    out.println("<li><a href=\"#Time\">Time</a>");
    out.println("</ul>");

    // Problem header
    if (!header.isEmpty()) {
      if (header.get(header.size() - 1).isEmpty()) {
        header.remove(header.size() - 1);
      }
      out.println("<h1 id=\"Problem-header\">Problem header</h1>");
      out.println("<pre>");
      for (var s : header) {
        wrap(s, out);
      }
      out.println("</pre>");
    }

    // Result
    out.println("<h1 id=\"Result\">Result</h1>");
    out.println("<table class=\"bordered\">");
    if (expected != null) {
      out.println("<tr>");
      out.println("<td class=\"bordered\">Expected");
      out.println("<td class=\"bordered\">" + expected);
    }
    out.println("<tr>");
    out.println("<td class=\"bordered\">Result");
    out.printf("<td class=\"bordered\"><b>%s</b>\n", result);
    out.println("</table>");

    // Proof
    if (refutation != null) {
      out.println("<h1 id=\"Proof\">Proof</h1>");
      out.println("<code>");
      out.println("</code>");
    }

    // Flush output
    out.close();
  }

  private static void wrap(String s, PrintWriter out) {
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
        out.println();
        column = 0;
      }

      // Print word
      var t = s.substring(i, j).replace("<", "&lt;");
      out.print(t);
      column += t.length();
      i = j;
    }
    out.println();
  }
}
