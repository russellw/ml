package prover;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public final class Main {
  private static ArrayList<String> files = new ArrayList<>();
  public static PrintStream writer;
  private static Language language;
  private static long timeout = 60_000;
  private static final String STDIN = "stdin";

  private static String optArg(String[] args, int i) throws IOException {
    if (i + 1 >= args.length) throw new IOException(args[i] + ": argument expected");
    return args[i + 1];
  }

  private static void args(String[] args) throws IOException {
    for (var i = 0; i < args.length; i++) {
      var arg = args[i];
      if (arg.isBlank()) continue;
      switch (arg.charAt(0)) {
        case '#':
          continue;
        case '-':
          break;
        default:
          if ("lst".equals(Etc.extension(arg))) {
            args(Files.readAllLines(Path.of(arg), StandardCharsets.UTF_8).toArray(new String[0]));
            continue;
          }
          files.add(arg);
          continue;
      }
      if ("-".equals(arg)) {
        files.add(STDIN);
        continue;
      }
      var opt = arg;
      while (opt.charAt(0) == '-') {
        opt = opt.substring(1);
      }
      String optArg = null;
      var j = 0;
      while ((j < opt.length()) && (Character.isLetter(opt.charAt(j)) || (opt.charAt(j) == '-'))) {
        j++;
      }
      if (j < opt.length()) {
        switch (opt.charAt(j)) {
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7':
          case '8':
          case '9':
            optArg = opt.substring(j);
            opt = opt.substring(0, j);
            break;
          case ':':
          case '=':
            optArg = opt.substring(j + 1);
            opt = opt.substring(0, j);
            break;
        }
      }
      switch (opt) {
        case "?":
        case "h":
        case "help":
          help();
          throw new IllegalStateException();
        case "T":
          {
            if (optArg == null) optArg = optArg(args, i++);
            var seconds = Double.parseDouble(optArg);
            new Timer()
                .schedule(
                    new TimerTask() {
                      @Override
                      public void run() {
                        System.exit(1);
                      }
                    },
                    (long) (seconds * 1000));
            break;
          }
        case "V":
        case "show-version":
        case "showversion":
        case "version":
          {
            var version = version();
            System.out.printf(
                "Prover %s, %s\n",
                Objects.toString(version, "[unknown version, not running from jar]"),
                System.getProperty("java.class.path"));
            System.out.printf(
                "%s, %s, %s\n",
                System.getProperty("java.vm.name"),
                System.getProperty("java.vm.version"),
                System.getProperty("java.home"));
            System.out.printf(
                "%s, %s, %s\n",
                System.getProperty("os.name"),
                System.getProperty("os.version"),
                System.getProperty("os.arch"));
            System.exit(0);
            throw new IllegalStateException();
          }
        case "dimacs":
          language = Language.DIMACS;
          break;
        case "t":
          {
            if (optArg == null) optArg = optArg(args, i++);
            var seconds = Double.parseDouble(optArg);
            timeout = (long) (seconds * 1000);
            break;
          }
        case "tptp":
          language = Language.TPTP;
          break;
        default:
          throw new IOException(arg + ": unknown option");
      }
    }
  }

  private Main() {}

  public static void main(String[] args) throws IOException {
    // Command line
    args(args);

    // Reports
    new File("logs").mkdir();

    // Statistics
    var solved = 0;

    // For each problem
    System.out.println("file                                     clauses sat processed   time");
    for (var file : files) {
      System.out.printf("%-40s", file);

      // Read
      var problem = TptpParser.read(file);
      System.out.printf(" %7d ", problem.clauses.size());

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
      var result = Superposition.satisfiable(problem.clauses, timeout);
      writer.close();

      // Result
      System.out.print(result == SZS.Timeout ? "   " : result.abbreviation());
      if (result.solved()) {
        if (!result.compatible(problem.expected))
          throw new IllegalStateException(result + " != " + problem.expected);
        solved++;
      }

      // Statistics
      System.out.printf(
          " %9d %6d\n", Superposition.processed.size(), System.currentTimeMillis() - start);
    }

    // Statistics
    System.out.printf(
        "solved %d/%d (%f%%)\n", solved, files.size(), solved * 100 / (double) files.size());
  }

  private static void help() {
    System.out.println("Usage: prover [options] file");
    System.out.println();
    System.out.println("General options:");
    System.out.println("-help       Show help");
    System.out.println("-version    Show version");
    System.out.println();
    System.out.println("Input:");
    System.out.println("-dimacs     DIMACS format");
    System.out.println("-tptp       TPTP format");
    System.out.println("-           Read from stdin");
    System.out.println();
    System.out.println("Resources:");
    System.out.println("-T seconds  Hard timeout");
    System.out.println("-t seconds  Soft timeout");
    System.out.println("            Seconds can be floating point");
    System.exit(0);
  }

  private static String version() throws IOException {
    var properties = new Properties();
    var stream =
        Main.class
            .getClassLoader()
            .getResourceAsStream("META-INF/maven/prover/prover/pom.properties");
    if (stream == null) {
      return null;
    }
    properties.load(stream);
    return properties.getProperty("version");
  }
}
