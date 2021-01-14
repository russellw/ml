package prover;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.NumberFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

public final class Summary {
  public final String name;
  public final int formulas;
  public final int clauses;
  public final SZS expected;
  public final SZS result;

  public Summary(Problem problem) {
    this.name = Etc.withoutExtension(Etc.withoutDir(problem.file()));
    this.formulas = problem.formulas.size();
    this.clauses = problem.clauses.size();
    this.expected = problem.expected;
    this.result = problem.result;
  }

  public static void write(String name, List<Summary> summaries) throws FileNotFoundException {
    var writer =
        new PrintWriter(
            Main.logDir
                + '/'
                + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HHmmss"))
                + ".html");
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

    // Flush output
    writer.close();
  }
}
