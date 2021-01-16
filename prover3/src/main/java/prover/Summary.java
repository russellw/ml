package prover;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.NumberFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Comparator;
import java.util.List;

public final class Summary {
  public final String name;
  public final int formulas;
  public final int clauses;
  public final int active;
  public final int passive;
  public final SZS expected;
  public final SZS result;
  public final double rating;
  public final long startTime;
  public final long endTime;

  public Summary(Problem problem) {
    name = Etc.baseName(problem.file());
    formulas = problem.formulas.size();
    clauses = problem.clauses.size();
    active = problem.superposition.active.size();
    passive = problem.superposition.passive.size();
    expected = problem.expected;
    result = problem.result;
    rating = problem.rating;
    startTime = problem.startTime;
    endTime = problem.endTime;
  }

  public static void write(String name, List<Summary> summaries) throws FileNotFoundException {
    if (summaries.isEmpty()) return;
    var now = LocalDateTime.now();
    var writer =
        new PrintWriter(
            Main.logDir
                + '/'
                + now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HHmmss"))
                + ".html");

    var numberFormat = NumberFormat.getInstance();
    var time = summaries.get(summaries.size() - 1).endTime - summaries.get(0).startTime;

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

    // Summaries
    summaries.sort(Comparator.comparingDouble((Summary o) -> o.rating).thenComparing(o -> o.name));
    writer.println("<table class=\"bordered\">");
    writer.println("<tr>");
    writer.println("<th class=\"bordered\">Problem");
    writer.println("<th class=\"bordered\">Formulas");
    writer.println("<th class=\"bordered\">Clauses");
    writer.println("<th class=\"bordered\">Active");
    writer.println("<th class=\"bordered\">Passive");
    writer.println("<th class=\"bordered\">Expected");
    writer.println("<th class=\"bordered\">Result");
    writer.println("<th class=\"bordered\">Solved");
    writer.println("<th class=\"bordered\">Rating");
    writer.println("<th class=\"bordered\">Time");
    for (var summary : summaries) {
      writer.println("<tr>");

      writer.print("<td class=\"bordered\">");
      writer.printf("<a href=\"%s.html\">%s</a>", summary.name, summary.name);

      writer.print("<td class=\"bordered\" style=\"text-align: right\">");
      if (summary.formulas > 0) writer.println(numberFormat.format(summary.formulas));

      writer.print("<td class=\"bordered\" style=\"text-align: right\">");
      writer.println(numberFormat.format(summary.clauses));

      writer.print("<td class=\"bordered\" style=\"text-align: right\">");
      writer.println(numberFormat.format(summary.active));

      writer.print("<td class=\"bordered\" style=\"text-align: right\">");
      writer.println(numberFormat.format(summary.passive));

      writer.print("<td class=\"bordered\">");
      if (summary.expected != null) writer.println(summary.expected);

      writer.print("<td class=\"bordered\">");
      writer.println(summary.result);

      writer.print("<td class=\"bordered\" style=\"text-align: center\">");
      if (Problem.solved(summary.result)) writer.print("&#x2714;");
      writer.println();

      writer.print("<td class=\"bordered\" style=\"text-align: right\">");
      if (summary.rating >= 0) writer.printf("%.2f", summary.rating);
      writer.println();

      writer.printf(
          "<td class=\"bordered\" style=\"text-align: right\">%.3f\n",
          (summary.endTime - summary.startTime) * 0.001);
    }
    writer.println("</table>");

    // Overall statistics
    writer.print("<p>");
    var solved = Etc.count(summaries, summary -> Problem.solved(summary.result));
    writer.printf(
        "Solved %d/%d (%f%%)\n",
        solved, summaries.size(), solved * 100 / (double) summaries.size());

    // Time
    writer.print("<p>");
    writer.printf("%.3f seconds\n", time * 0.001);
    writer.print("<p>");
    writer.println(now.format(DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy, HH:mm:ss")));

    // Flush output
    writer.close();
  }
}
