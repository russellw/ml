package prover;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

public final class Problem {
  public final String file;
  public ArrayList<String> header = new ArrayList<>();
  public SZS expected;
  public ArrayList<Formula> formulas = new ArrayList<>();
  public Formula conjecture;
  public ArrayList<Clause> clauses = new ArrayList<>();
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

  public void write() throws FileNotFoundException {
    // Report
    new File("logs").mkdir();
    var writer = new PrintStream("logs/" + new File(file).getName().split("\\.")[0] + ".html");
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
    writer.println("<li><a href=\"#Memory\">Memory</a>");
    writer.println("<li><a href=\"#Time\">Time</a>");
    writer.println("</ul>");

    writer.close();
  }
}
