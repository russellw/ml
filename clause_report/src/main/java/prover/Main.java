package prover;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class Main {
  public static Boolean status;
  public static PrintStream writer;

  public static void main(String[] args) throws IOException {
    // Command line
    var arg = args[0];
    String[] files;
    if (arg.endsWith(".lst"))
      files = Files.readAllLines(Path.of(arg), StandardCharsets.UTF_8).toArray(new String[0]);
    else files = new String[] {arg};

    // Reports
    new File("logs").mkdir();

    // Statistics
    var solved = 0;

    // For each problem
    System.out.println("file                                     clauses sat   processed   time");
    for (var file : files) {
      System.out.printf("%-40s", file);

      // Read
      status = null;
      var clauses = TptpParser.read(file);
      System.out.printf(" %7d", clauses.size());

      // Report
      writer = new PrintStream("logs/" + new File(file).getName().split("\\.")[0] + ".html");
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

      // Solve
      var start = System.currentTimeMillis();
      Superposition.timeout = start + 3_000;
      var result = Superposition.satisfiable(clauses);
      writer.close();

      // Result
      if (result == null) System.out.print("      ");
      else {
        System.out.print(result ? " sat  " : " unsat");
        if (result != status) throw new IllegalStateException();
        solved++;
      }

      // Statistics
      System.out.printf(
          " %9d %6d\n", Superposition.processed.size(), System.currentTimeMillis() - start);
    }

    // Statistics
    System.out.printf(
        "solved %d/%d (%f%%)\n", solved, files.length, solved * 100 / (double) files.length);
  }
}
