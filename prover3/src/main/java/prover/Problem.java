package prover;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
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
    var writer = new PrintStream("logs/" + Etc.removeExtension(Etc.removeDir(file)) + ".html");
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
    if (Main.version() != null) writer.println("<li><a href=\"#Version\">Version</a>");
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

    // Flush output
    writer.close();
  }

  private void wrap(String s, PrintStream writer) {
    var col = 0;
    for (var i = 0; i < s.length(); ) {
      var j = i;
      while ((j < s.length()) && (s.charAt(j) == ' ')) {
        j++;
      }
      var word = j;
      while ((j < s.length()) && (s.charAt(j) != ' ')) {
        j++;
      }
      j = Math.min(j, word + 90);
      if ((col > 0) && (col + (j - i) > 80)) {
        while ((i < j) && (s.charAt(i) == ' ')) {
          i++;
        }
        writer.println();
        col = 0;
      }
      writer.print(s.substring(i, j).replace("<", "&lt;"));
      col += j - i;
      i = j;
    }
    writer.println();
  }
}
